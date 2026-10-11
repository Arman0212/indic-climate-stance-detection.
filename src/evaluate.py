"""Evaluation module computing Macro F1, Accuracy, per-class metrics, confusion matrix, and logging."""

import argparse
import csv
import datetime
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_predictions(
    predictions_path: str,
    config_path: str = "config.yaml",
    model_name: str = "TF-IDF Char (1-5) + Logistic Regression",
    settings_str: str = "analyzer=char, ngrams=(1,5), C=1.0, class_weight=balanced",
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Calculate and display evaluation metrics, generate confusion matrix plot, and log results.

    Args:
        predictions_path: Path to CSV containing true and predicted labels.
        config_path: Path to configuration YAML.
        model_name: Name of model for logging.
        settings_str: Hyperparameter settings string for logging.
        output_dir: Directory for confusion matrix and results log.

    Returns:
        Dictionary of computed summary metrics.
    """
    config = load_config(config_path)
    cols = config.get("data", {}).get("columns", {})
    true_col = cols.get("label", "label")
    pred_col = "pred_label"

    df = pd.read_csv(predictions_path, keep_default_na=False)
    if true_col not in df.columns:
        # Fall back to alternative naming if needed
        if "true_label" in df.columns:
            true_col = "true_label"
        else:
            raise ValueError(
                f"Evaluation file {predictions_path} missing true label column. Found: {list(df.columns)}"
            )

    if pred_col not in df.columns:
        if "predicted_label" in df.columns:
            pred_col = "predicted_label"
        else:
            raise ValueError(
                f"Evaluation file {predictions_path} missing predicted label column. Found: {list(df.columns)}"
            )

    y_true = df[true_col].astype(str)
    y_pred = df[pred_col].astype(str)

    target_classes = config.get("labels", {}).get("classes", ["Favour", "Against", "None"])
    # Filter or reorder classes present in data
    present_classes = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
    class_order = [c for c in target_classes if c in present_classes]
    for c in present_classes:
        if c not in class_order:
            class_order.append(c)

    # Core metrics
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    clf_report_dict = classification_report(
        y_true, y_pred, labels=class_order, output_dict=True, zero_division=0
    )
    clf_report_text = classification_report(
        y_true, y_pred, labels=class_order, digits=4, zero_division=0
    )

    # Per-fold macro F1 if fold column exists
    per_fold_f1s: List[float] = []
    if "fold" in df.columns and df["fold"].nunique() > 1:
        folds = sorted(df["fold"].unique())
        for f in folds:
            fold_df = df[df["fold"] == f]
            fold_f1 = f1_score(
                fold_df[true_col], fold_df[pred_col], average="macro", zero_division=0
            )
            per_fold_f1s.append(float(fold_f1))

    mean_fold_f1 = float(np.mean(per_fold_f1s)) if per_fold_f1s else float(macro_f1)

    # Output directory preparation
    out_dir_path = Path(
        output_dir or config.get("outputs", {}).get("output_dir", "outputs")
    )
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred, labels=class_order)
    cm_path = Path(
        config.get("outputs", {}).get(
            "confusion_matrix_path", str(out_dir_path / "confusion_matrix.png")
        )
    )
    cm_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_order,
        yticklabels=class_order,
        cbar=False,
    )
    plt.title(f"Confusion Matrix: {model_name}\n(Macro F1: {macro_f1:.4f})", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info("Saved confusion matrix heatmap to %s", cm_path)

    # Append to results_log.csv
    log_path = Path(
        config.get("outputs", {}).get(
            "results_log_path", str(out_dir_path / "results_log.csv")
        )
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.is_file()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    per_fold_str = ";".join(f"{score:.4f}" for score in per_fold_f1s) if per_fold_f1s else f"{macro_f1:.4f}"

    fieldnames = [
        "timestamp",
        "model_name",
        "settings",
        "per_fold_macro_f1",
        "mean_macro_f1",
        "overall_macro_f1",
        "accuracy",
    ]

    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp,
                "model_name": model_name,
                "settings": settings_str,
                "per_fold_macro_f1": per_fold_str,
                "mean_macro_f1": f"{mean_fold_f1:.4f}",
                "overall_macro_f1": f"{macro_f1:.4f}",
                "accuracy": f"{accuracy:.4f}",
            }
        )
    logger.info("Appended benchmark record to %s", log_path)

    # Print comprehensive report to console
    print("\n" + "=" * 65)
    print(f" EVALUATION REPORT: {model_name}")
    print("=" * 65)
    print(f"Overall Macro F1 : {macro_f1:.4f}")
    print(f"Overall Accuracy : {accuracy * 100:.2f}%")
    if per_fold_f1s:
        folds_str = ", ".join(f"Fold {i}: {s:.4f}" for i, s in enumerate(per_fold_f1s))
        print(f"5-Fold Macro F1s : [{folds_str}]")
        print(f"Mean Fold F1     : {mean_fold_f1:.4f}")
    print("-" * 65)
    print("Per-Class Breakdown:")
    print(clf_report_text)
    print(f"Confusion Matrix saved to: {cm_path.resolve()}")
    print("=" * 65 + "\n")

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "mean_fold_f1": mean_fold_f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate stance detection predictions and log metrics."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/oof_predictions.csv",
        help="Path to CSV with true and predicted labels (default: outputs/oof_predictions.csv)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to project config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="TF-IDF Char (1-5) + Logistic Regression",
        help="Model identifier string for results log",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="analyzer=char, ngrams=(1,5), C=1.0, class_weight=balanced",
        help="Hyperparameter description string for results log",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save logs and plots (default: outputs/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_predictions(
        predictions_path=args.input,
        config_path=args.config,
        model_name=args.model_name,
        settings_str=args.settings,
        output_dir=args.output_dir,
    )
