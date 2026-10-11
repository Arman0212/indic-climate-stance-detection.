"""Prediction module to generate predicted stance labels for unlabelled or test comments."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import clean_text
from src.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def predict_stances(
    input_path: str,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    config_path: str = "config.yaml",
) -> str:
    """Load trained pipeline, predict stance on input CSV, and save predictions.

    Args:
        input_path: Path to input test CSV (must contain 'id' and 'text').
        output_path: Destination path for predictions CSV.
        model_path: Path to serialized trained model joblib.
        config_path: Path to configuration YAML.

    Returns:
        Path to output predictions CSV.
    """
    config = load_config(config_path)
    cols = config.get("data", {}).get("columns", {})
    id_col = cols.get("id", "id")
    text_col = cols.get("text", "text")

    m_path = Path(
        model_path
        or config.get("outputs", {}).get("model_path", "outputs/baseline_model.joblib")
    )
    if not m_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {m_path.resolve()}. "
            "Please train the model first by running src/train_baseline.py."
        )

    logger.info("Loading model from %s", m_path)
    model = joblib.load(m_path)

    df = pd.read_csv(input_path, keep_default_na=False)
    if id_col not in df.columns or text_col not in df.columns:
        raise ValueError(
            f"Input CSV {input_path} must contain '{id_col}' and '{text_col}'. Found: {list(df.columns)}"
        )

    logger.info("Running inference on %d comments from %s", len(df), input_path)
    # Apply standard cleaning before feature extraction
    cleaned_texts = df[text_col].fillna("").astype(str).apply(clean_text)
    predictions = model.predict(cleaned_texts)

    out_df = pd.DataFrame(
        {
            "id": df[id_col],
            "predicted_label": predictions,
        }
    )

    out_file = Path(
        output_path
        or config.get("outputs", {}).get("predictions_path", "outputs/predictions.csv")
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_file, index=False)

    logger.info("Saved predictions to %s", out_file)
    return str(out_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict stance labels (Favour, Against, None) using trained model."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV to predict on (default: sample_comments.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output predictions CSV (default: outputs/predictions.csv)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to saved model (default: outputs/baseline_model.joblib)",
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
    predict_stances(
        input_path=in_file,
        output_path=args.output,
        model_path=args.model_path,
        config_path=args.config,
    )
