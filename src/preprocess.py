"""Text preprocessing module for Hindi and Bengali social media comments."""

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Regular expressions for URL and @mention removal
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Preprocess and normalize noisy social media text.

    Current pipeline operations:
    1. Unicode NFC normalization.
    2. URL removal.
    3. @mention removal.
    4. Whitespace collapse.

    Future extension hooks are clearly demarcated below.

    Args:
        text: Raw comment string.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Unicode NFC normalization (ensures canonical combining characters)
    text = unicodedata.normalize("NFC", text)

    # 2. URL removal
    text = URL_PATTERN.sub(" ", text)

    # 3. @mention removal
    text = MENTION_PATTERN.sub(" ", text)

    # =========================================================================
    # TODO [HOOK 1: Emoji Handling]
    # In future iterations, handle or preserve domain-relevant emojis:
    # - Option A: Convert emojis to textual descriptions using emoji.demojize(text)
    # - Option B: Map climate emojis (e.g., 🌱 -> 'nature', 🔥 -> 'crisis')
    # - Option C: Strip non-sentiment emojis while retaining emotion cues
    # =========================================================================

    # =========================================================================
    # TODO [HOOK 2: Elongation Collapse]
    # In future iterations, collapse repeated character sequences common in
    # informal comments (e.g., "bahaaaat" -> "bahat", "soooo" -> "soo",
    # or repeated Indic punctuation/matras like '?????' or '!!!!!').
    # =========================================================================

    # =========================================================================
    # TODO [HOOK 3: Transliteration & Script Normalization]
    # In future iterations, transliterate Romanized Hindi/Bengali text into
    # native script using indic_transliteration.sanscript or normalize
    # dialectal spelling variants with Indic NLP Library.
    # =========================================================================

    # 4. Whitespace collapse and trim
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def preprocess_file(
    input_path: str,
    output_path: str,
    text_col: str = "text",
) -> pd.DataFrame:
    """Read a CSV file, apply clean_text to the specified column, and save output.

    Args:
        input_path: Path to input CSV.
        output_path: Destination path for cleaned CSV.
        text_col: Name of text column to clean.

    Returns:
        Processed DataFrame.
    """
    df = pd.read_csv(input_path, keep_default_na=False)
    if text_col not in df.columns:
        raise ValueError(
            f"Input file {input_path} missing text column '{text_col}'. Columns: {list(df.columns)}"
        )

    logger.info("Cleaning %d rows from %s (column: '%s')", len(df), input_path, text_col)
    df[text_col] = df[text_col].apply(clean_text)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    logger.info("Saved preprocessed data to %s", out_file)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess comments (Unicode NFC, URL/@mention removal, whitespace collapse)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV file path (default: sample_path from config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: processed_sample_path from config.yaml)",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=None,
        help="Column name containing text to preprocess",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to project config.yaml (default: config.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    in_file = args.input or config.get("data", {}).get(
        "sample_path", "data/sample/sample_comments.csv"
    )
    out_file = args.output or config.get("data", {}).get(
        "processed_sample_path", "data/processed/sample_preprocessed.csv"
    )
    col = args.text_col or config.get("data", {}).get("columns", {}).get("text", "text")

    preprocess_file(input_path=in_file, output_path=out_file, text_col=col)
