"""YouTube comment scraper for Hindi and Bengali climate change videos.

Uses YouTube Data API v3 to search for videos and fetch top-level comments,
identifying language by script and writing stand-in data to data/raw/.
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

# Ensure local imports work when run as script or module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import guess_language, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_youtube_client(api_key: Optional[str] = None):
    """Instantiate and return the YouTube Data API v3 client.

    Args:
        api_key: YouTube API key. If None, reads from YOUTUBE_API_KEY in environment.

    Returns:
        Google API client service object.
    """
    load_dotenv()
    key = api_key or os.getenv("YOUTUBE_API_KEY")

    if not key or key.strip() in ("", "your_youtube_api_key_here"):
        raise ValueError(
            "YOUTUBE_API_KEY environment variable is missing or unset. "
            "Please create a .env file based on .env.example with your YouTube API key."
        )

    try:
        from googleapiclient.discovery import build

        return build("youtube", "v3", developerKey=key)
    except Exception as e:
        logger.error("Failed to build YouTube API client: %s", e)
        raise


def is_comments_disabled_error(error: Exception) -> bool:
    """Check if an HttpError indicates comments are disabled."""
    from googleapiclient.errors import HttpError

    if isinstance(error, HttpError):
        error_content = str(error)
        return "commentsDisabled" in error_content or error.resp.status == 403 and "forbidden" in error_content
    return False


def is_quota_exceeded_error(error: Exception) -> bool:
    """Check if an HttpError indicates API quota has been exceeded."""
    from googleapiclient.errors import HttpError

    if isinstance(error, HttpError):
        error_content = str(error)
        return (
            "quotaExceeded" in error_content
            or "dailyLimitExceeded" in error_content
            or (error.resp.status == 403 and "quota" in error_content.lower())
        )
    return False


def fetch_video_ids_for_query(
    youtube, query: str, max_videos: int = 25
) -> List[str]:
    """Search YouTube for videos matching a query string.

    Args:
        youtube: YouTube API client.
        query: Search term.
        max_videos: Maximum number of video IDs to retrieve.

    Returns:
        List of video ID strings.
    """
    video_ids: List[str] = []
    page_token = None

    while len(video_ids) < max_videos:
        batch_size = min(50, max_videos - len(video_ids))
        try:
            req = youtube.search().list(
                q=query,
                part="id",
                type="video",
                maxResults=batch_size,
                pageToken=page_token,
            )
            resp = req.execute()
        except Exception as e:
            if is_quota_exceeded_error(e):
                logger.error("Quota exceeded while searching videos: %s", e)
                raise
            logger.warning("Error searching videos for query '%s': %s", query, e)
            break

        for item in resp.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if vid and vid not in video_ids:
                video_ids.append(vid)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    logger.info("Found %d videos for query '%s'", len(video_ids), query)
    return video_ids


def fetch_comments_for_video(
    youtube,
    video_id: str,
    query: str,
    seen_ids: Set[str],
    max_comments_per_video: int = 100,
) -> List[Dict[str, str]]:
    """Retrieve top-level comments for a given video ID.

    Args:
        youtube: YouTube API client.
        video_id: YouTube video ID.
        query: Associated search query.
        seen_ids: Set of comment IDs already collected (for deduplication).
        max_comments_per_video: Max comments to pull from this video.

    Returns:
        List of comment dictionaries with keys (id, text, video_id, query, lang_guess).
    """
    comments: List[Dict[str, str]] = []
    page_token = None

    while len(comments) < max_comments_per_video:
        batch_size = min(100, max_comments_per_video - len(comments))
        try:
            req = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=batch_size,
                textFormat="plainText",
                pageToken=page_token,
            )
            resp = req.execute()
        except Exception as e:
            if is_comments_disabled_error(e):
                logger.debug("Comments disabled for video %s", video_id)
                return comments
            if is_quota_exceeded_error(e):
                logger.error("API quota exceeded while fetching comments: %s", e)
                raise
            logger.warning("Error fetching comments for video %s: %s", video_id, e)
            return comments

        for item in resp.get("items", []):
            top_level = item.get("snippet", {}).get("topLevelComment", {})
            cid = top_level.get("id")
            snippet = top_level.get("snippet", {})
            raw_text = snippet.get("textOriginal") or snippet.get("textDisplay") or ""

            if not cid or cid in seen_ids:
                continue

            clean_display = raw_text.strip()
            if not clean_display:
                continue

            seen_ids.add(cid)
            lang = guess_language(clean_display)

            comments.append(
                {
                    "id": cid,
                    "text": clean_display,
                    "video_id": video_id,
                    "query": query,
                    "lang_guess": lang,
                }
            )

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return comments


def scrape_youtube(
    config_path: str = "config.yaml",
    output_path: Optional[str] = None,
    target_per_language: Optional[int] = None,
    queries: Optional[List[str]] = None,
) -> str:
    """Scrape comments across search queries until target per language is reached.

    Args:
        config_path: Path to YAML config file.
        output_path: Destination CSV path.
        target_per_language: Target comments per language (Hindi, Bengali).
        queries: List of search queries.

    Returns:
        Output CSV filepath.
    """
    config = load_config(config_path)
    yt_cfg = config.get("youtube", {})
    queries_to_run = queries or yt_cfg.get(
        "queries", ["जलवायु परिवर्तन", "ग्लोबल वार्मिंग", "জলবায়ু পরিবর্তন", "বৈশ্বিক উষ্ণতা"]
    )
    target = target_per_language or yt_cfg.get("target_per_language", 500)
    out_file = output_path or config.get("data", {}).get(
        "standin_comments_path", "data/raw/standin_comments.csv"
    )

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    youtube = build_youtube_client()

    seen_ids: Set[str] = set()
    collected_rows: List[Dict[str, str]] = []
    lang_counts: Dict[str, int] = {"hindi": 0, "bengali": 0, "other": 0}

    logger.info("Starting YouTube scraping with target of %d comments per language", target)

    quota_hit = False
    for query in queries_to_run:
        if lang_counts["hindi"] >= target and lang_counts["bengali"] >= target:
            logger.info("Reached target comment count for both Hindi and Bengali.")
            break

        logger.info("Processing search query: '%s'", query)
        try:
            video_ids = fetch_video_ids_for_query(youtube, query=query, max_videos=20)
        except Exception as e:
            if is_quota_exceeded_error(e):
                logger.error("Quota exceeded during query search. Stopping scrape.")
                quota_hit = True
                break
            continue

        for vid in video_ids:
            if lang_counts["hindi"] >= target and lang_counts["bengali"] >= target:
                break
            try:
                comments = fetch_comments_for_video(
                    youtube, video_id=vid, query=query, seen_ids=seen_ids
                )
                for comment in comments:
                    collected_rows.append(comment)
                    lang = comment["lang_guess"]
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
            except Exception as e:
                if is_quota_exceeded_error(e):
                    logger.error("Quota exceeded during comment fetching. Stopping scrape.")
                    quota_hit = True
                    break
                logger.warning("Skipping video %s due to error: %s", vid, e)

        if quota_hit:
            break

    # Write collected comments to CSV
    fieldnames = ["id", "text", "video_id", "query", "lang_guess"]
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(collected_rows)

    logger.info(
        "Scraping completed. Saved %d total comments to %s (Hindi: %d, Bengali: %d, Other: %d)",
        len(collected_rows),
        out_file,
        lang_counts.get("hindi", 0),
        lang_counts.get("bengali", 0),
        lang_counts.get("other", 0),
    )
    return out_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape YouTube comments for Hindi and Bengali climate queries."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to project config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV (default: from config.yaml)",
    )
    parser.add_argument(
        "--target-per-lang",
        type=int,
        default=None,
        help="Target comments per language (default: 500 or config value)",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        help="List of search queries override",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scrape_youtube(
        config_path=args.config,
        output_path=args.output,
        target_per_language=args.target_per_lang,
        queries=args.queries,
    )
