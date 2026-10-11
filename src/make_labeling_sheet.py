"""Create labeling sheet for manual annotation and compute inter-annotator agreement."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def sample_for_labeling(
    input_path: str,
    output_path: str,
    n_per_lang: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample comments stratified by language for human annotation.

    Args:
        input_path: Path to raw comments CSV (expects 'id', 'text', 'lang_guess').
        output_path: Destination path for the labeling sheet.
        n_per_lang: Number of comments to sample per language (Hindi, Bengali).
        seed: Random seed for sampling.

    Returns:
        Sampled DataFrame with empty annotator columns.
    """
    set_seed(seed)
    df = pd.read_csv(input_path, keep_default_na=False)

    if "lang_guess" not in df.columns:
        if "text" in df.columns:
            logger.info("Column 'lang_guess' not found; automatically guessing language from text...")
            from src.utils import guess_language
            df["lang_guess"] = df["text"].astype(str).apply(guess_language)
        else:
            raise ValueError(
                f"Input CSV {input_path} must contain 'lang_guess' or 'text' column. Found: {list(df.columns)}"
            )

    hindi_df = df[df["lang_guess"].str.lower() == "hindi"]
    bengali_df = df[df["lang_guess"].str.lower() == "bengali"]

    n_hindi = min(n_per_lang, len(hindi_df))
    n_bengali = min(n_per_lang, len(bengali_df))

    if n_hindi < n_per_lang:
        logger.warning(
            "Requested %d Hindi comments, but only %d available in %s",
            n_per_lang,
            len(hindi_df),
            input_path,
        )
    if n_bengali < n_per_lang:
        logger.warning(
            "Requested %d Bengali comments, but only %d available in %s",
            n_per_lang,
            len(bengali_df),
            input_path,
        )

    sampled_hindi = (
        hindi_df.sample(n=n_hindi, random_state=seed) if n_hindi > 0 else hindi_df
    )
    sampled_bengali = (
        bengali_df.sample(n=n_bengali, random_state=seed) if n_bengali > 0 else bengali_df
    )

    combined = pd.concat([sampled_hindi, sampled_bengali], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Add empty annotator columns
    combined["label_annotator1"] = ""
    combined["label_annotator2"] = ""

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_file, index=False)

    logger.info(
        "Successfully sampled %d total comments (%d Hindi, %d Bengali) into %s",
        len(combined),
        n_hindi,
        n_bengali,
        out_file,
    )
    return combined


def compute_agreement(sheet_path: str) -> None:
    """Read annotated sheet and report agreement rate, Cohen's kappa, and disagreements.

    Args:
        sheet_path: Path to CSV with columns 'label_annotator1' and 'label_annotator2'.
    """
    df = pd.read_csv(sheet_path, keep_default_na=False)

    col1 = "label_annotator1"
    col2 = "label_annotator2"

    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(
            f"Annotation sheet must contain '{col1}' and '{col2}' columns. Found: {list(df.columns)}"
        )

    # Filter out unannotated rows (keeping literal 'None' stance label)
    s1 = df[col1].astype(str).str.strip()
    s2 = df[col2].astype(str).str.strip()
    valid_mask = (s1 != "") & (s2 != "")
    annotated = df[valid_mask].copy()
    annotated[col1] = s1[valid_mask]
    annotated[col2] = s2[valid_mask]

    if len(annotated) == 0:
        logger.warning(
            "No doubly-annotated rows found in %s. Please enter labels in %s and %s.",
            sheet_path,
            col1,
            col2,
        )
        return

    total = len(annotated)
    agreements = (annotated[col1] == annotated[col2]).sum()
    agreement_rate = (agreements / total) * 100.0
    kappa = cohen_kappa_score(annotated[col1], annotated[col2])

    print("\n" + "=" * 60)
    print(" INTER-ANNOTATOR AGREEMENT REPORT")
    print("=" * 60)
    print(f"Total doubly-annotated samples : {total}")
    print(f"Exact Agreement Count          : {agreements} / {total}")
    print(f"Raw Agreement Rate             : {agreement_rate:.2f}%")
    print(f"Cohen's Kappa (κ)              : {kappa:.4f}")
    print("=" * 60)

    disagreements = annotated[annotated[col1] != annotated[col2]]
    if len(disagreements) > 0:
        print(f"\nDisagreements ({len(disagreements)} rows):")
        print("-" * 60)
        for _, row in disagreements.iterrows():
            cid = row.get("id", "N/A")
            text = str(row.get("text", ""))[:70].replace("\n", " ")
            a1 = row[col1]
            a2 = row[col2]
            print(f"ID: {cid:<12} | A1: {a1:<8} | A2: {a2:<8} | Text: {text}...")
        print("-" * 60)
    else:
        print("\nPerfect agreement! No conflicting annotations found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create labeling sheet or evaluate inter-annotator agreement."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample", "agreement"],
        default="sample",
        help="Mode to run: 'sample' creates to_label.csv; 'agreement' evaluates filled sheet.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to project config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV file path override",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path override",
    )
    parser.add_argument(
        "--n-per-lang",
        type=int,
        default=50,
        help="Number of comments to sample per language in 'sample' mode (default: 50)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    seed = config.get("project", {}).get("seed", 42)

    if args.mode == "sample":
        in_file = args.input or config.get("data", {}).get(
            "standin_comments_path", "data/raw/standin_comments.csv"
        )
        out_file = args.output or config.get("data", {}).get(
            "to_label_path", "data/processed/to_label.csv"
        )
        sample_for_labeling(
            input_path=in_file,
            output_path=out_file,
            n_per_lang=args.n_per_lang,
            seed=seed,
        )
    elif args.mode == "agreement":
        sheet_file = args.input or config.get("data", {}).get(
            "to_label_path", "data/processed/to_label.csv"
        )
        compute_agreement(sheet_path=sheet_file)
