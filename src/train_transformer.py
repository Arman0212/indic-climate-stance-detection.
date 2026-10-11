"""Fine-tuning multilingual transformer models (e.g., IndicBERT, XLM-RoBERTa, mBERT).

This stub defines the planned interface for fine-tuning pre-trained transformer
encoders on Hindi and Bengali climate change stance detection.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_transformer(
    pretrained_model_name: str = "ai4bharat/indic-bert",
    config_path: str = "config.yaml",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Fine-tune a transformer model for 3-class stance detection.

    Planned Implementation Details:
    1. Load tokenizer and sequence classification model (e.g. AutoTokenizer,
       AutoModelForSequenceClassification from Hugging Face Transformers).
    2. Tokenize input text using max_length padding/truncation.
    3. Stratify batches across folds defined in data/processed/folds.csv.
    4. Train with cross-entropy loss (incorporating class weights to counter
       stance imbalance).
    5. Evaluate on validation folds and save model checkpoint and out-of-fold
       probabilities.

    Args:
        pretrained_model_name: Hugging Face repository or path (e.g.,
            'ai4bharat/indic-bert', 'xlm-roberta-base', 'bert-base-multilingual-cased').
        config_path: Path to configuration YAML.
        epochs: Number of training epochs.
        batch_size: Mini-batch size for training.
        learning_rate: AdamW optimizer learning rate.
        output_dir: Checkpoint and output directory.

    Returns:
        Dictionary of validation metrics.
    """
    config = load_config(config_path)
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    logger.info(
        "[STUB] train_transformer called with model '%s' (epochs=%d, lr=%e).",
        pretrained_model_name,
        epochs,
        learning_rate,
    )
    logger.info(
        "[STUB] To be implemented when official training dataset is released. "
        "Will load folds from %s and fine-tune using Hugging Face Trainer.",
        config.get("data", {}).get("folds_path", "data/processed/folds.csv"),
    )
    return {"status": "stub_ready", "macro_f1": 0.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune multilingual transformer models for stance detection (STUB)."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ai4bharat/indic-bert",
        help="Hugging Face model identifier (default: ai4bharat/indic-bert)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to project config.yaml",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_transformer(
        pretrained_model_name=args.model_name,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
