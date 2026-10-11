"""Train TF-IDF + Logistic Regression baseline model with Stratified 5-Fold Cross-Validation."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_baseline(
    config_path: str = "config.yaml",
    input_path: Optional[str] = None,
    folds_output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Execute stratified 5-fold CV and train final baseline classifier.

    Args:
        config_path: Path to configuration YAML.
        input_path: Path to preprocessed training data CSV.
        folds_output_path: Destination path for fold assignments CSV.
        output_dir: Directory to save out-of-fold probabilities and trained model.
    """
    config = load_config(config_path)
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    cols = config.get("data", {}).get("columns", {})
    id_col = cols.get("id", "id")
    text_col = cols.get("text", "text")
    label_col = cols.get("label", "label")

    in_file = input_path or config.get("data", {}).get(
        "processed_sample_path", "data/processed/sample_preprocessed.csv"
    )
    if not Path(in_file).is_file():
        # Fall back to sample path if preprocessed doesn't exist yet
        in_file = config.get("data", {}).get("sample_path", "data/sample/sample_comments.csv")

    df = pd.read_csv(in_file, keep_default_na=False)
    logger.info("Loaded %d training instances from %s", len(df), in_file)

    for required in (id_col, text_col, label_col):
        if required not in df.columns:
            raise ValueError(f"Input file {in_file} is missing required column '{required}'")

    # Ensure label column is clean string format (handles literal 'None' label)
    df[label_col] = df[label_col].astype(str).str.strip()

    # Hyperparameters
    train_cfg = config.get("training", {})
    n_splits = train_cfg.get("n_splits", 5)
    ngram_min = train_cfg.get("ngram_min", 1)
    ngram_max = train_cfg.get("ngram_max", 5)
    c_param = train_cfg.get("c_param", 1.0)
    max_iter = train_cfg.get("max_iter", 1000)

    # Class ordering
    classes = sorted(df[label_col].unique())
    logger.info("Detected classes (%d): %s", len(classes), classes)

    # Prepare directories
    out_dir = Path(output_dir or config.get("outputs", {}).get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    folds_file = Path(
        folds_output_path
        or config.get("data", {}).get("folds_path", "data/processed/folds.csv")
    )
    folds_file.parent.mkdir(parents=True, exist_ok=True)

    # Setup Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1

    oof_probs = np.zeros((len(df), len(classes)), dtype=np.float64)
    oof_preds = [""] * len(df)

    logger.info("Starting %d-fold Stratified Cross-Validation...", n_splits)

    for fold_idx, (train_indices, val_indices) in enumerate(
        skf.split(df[text_col], df[label_col])
    ):
        df.loc[val_indices, "fold"] = fold_idx

        train_texts = df.loc[train_indices, text_col].fillna("").astype(str)
        train_labels = df.loc[train_indices, label_col]
        val_texts = df.loc[val_indices, text_col].fillna("").astype(str)

        fold_pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(ngram_min, ngram_max),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        C=c_param,
                        max_iter=max_iter,
                        random_state=seed,
                    ),
                ),
            ]
        )

        fold_pipeline.fit(train_texts, train_labels)

        # Ensure probabilities match class index ordering
        fold_classes = list(fold_pipeline.classes_)
        val_probs = fold_pipeline.predict_proba(val_texts)

        # Align probability columns to global class order
        for c_idx, c_name in enumerate(classes):
            orig_idx = fold_classes.index(c_name)
            oof_probs[val_indices, c_idx] = val_probs[:, orig_idx]

        for i, val_idx in enumerate(val_indices):
            best_class_idx = np.argmax(oof_probs[val_idx])
            oof_preds[val_idx] = classes[best_class_idx]

    # Save fold assignments
    df.to_csv(folds_file, index=False)
    logger.info("Saved fold assignments to %s", folds_file)

    # Save OOF probabilities
    oof_probs_path = out_dir / "oof_probabilities.npy"
    np.save(oof_probs_path, oof_probs)
    logger.info("Saved out-of-fold probability matrix to %s", oof_probs_path)

    # Save detailed OOF predictions DataFrame
    oof_df = df[[id_col, text_col, label_col, "fold"]].copy()
    oof_df["pred_label"] = oof_preds
    for c_idx, c_name in enumerate(classes):
        oof_df[f"prob_{c_name}"] = oof_probs[:, c_idx]

    oof_preds_csv = out_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_preds_csv, index=False)
    logger.info("Saved detailed out-of-fold predictions to %s", oof_preds_csv)

    # Train final model on full dataset
    logger.info("Fitting final model on complete dataset (%d instances)...", len(df))
    final_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(ngram_min, ngram_max),
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    C=c_param,
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )
    final_pipeline.fit(df[text_col].fillna("").astype(str), df[label_col])

    model_path = Path(
        config.get("outputs", {}).get("model_path", "outputs/baseline_model.joblib")
    )
    joblib.dump(final_pipeline, model_path)
    logger.info("Saved final fitted model to %s", model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + Logistic Regression baseline model with 5-fold CV."
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
        "--folds-output",
        type=str,
        default=None,
        help="Destination path for folds.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory override (default: outputs/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_baseline(
        config_path=args.config,
        input_path=args.input,
        folds_output_path=args.folds_output,
        output_dir=args.output_dir,
    )
