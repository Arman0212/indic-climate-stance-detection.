"""Zero-shot and few-shot in-context learning prompting with Large Language Models.

This stub provides the blueprint for prompting LLMs (e.g., Gemini, LLaMA,
Claude) to detect stance toward the claim:
"Climate change and global warming is a serious concern."
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are an expert NLP annotator specializing in multilingual stance detection on South Asian social media (Hindi and Bengali comments, including Romanized text and code-mixing).

Target Claim: "Climate change and global warming is a serious concern."

Your task is to classify the author's stance toward this claim into exactly one of three categories:
1. Favour: The comment explicitly or implicitly agrees with the claim (acknowledges climate urgency, human impact, environmental damage, need to save nature/glaciers).
2. Against: The comment rejects, trivializes, or opposes the claim (claims global warming is a hoax, natural variation, Western conspiracy, or propaganda).
3. None: The comment is neutral, off-topic (e.g. praising video quality, music, asking unrelated questions), or ambiguous without clear stance.

Output only a valid JSON object with keys "id" and "label".
"""


def prompt_llm(
    input_path: str,
    output_path: Optional[str] = None,
    config_path: str = "config.yaml",
    prompt_mode: str = "few_shot",
    model_provider: str = "gemini",
) -> Dict[str, str]:
    """Execute LLM in-context prompting for stance classification.

    Planned Implementation Details:
    1. Construct prompt using system instructions and few-shot exemplars
       selected from each fold or representative cluster.
    2. Dispatch batch or async requests to LLM API (Google GenAI, OpenAI, etc.).
    3. Parse structured outputs, validate against ['Favour', 'Against', 'None'].
    4. Save predictions to outputs/predictions_llm.csv.

    Args:
        input_path: Path to comments CSV.
        output_path: Path to output predictions CSV.
        config_path: Path to configuration YAML.
        prompt_mode: 'zero_shot' or 'few_shot'.
        model_provider: LLM provider identifier.

    Returns:
        Status summary dictionary.
    """
    config = load_config(config_path)
    logger.info(
        "[STUB] prompt_llm called (provider='%s', mode='%s', input='%s').",
        model_provider,
        prompt_mode,
        input_path,
    )
    logger.info(
        "[STUB] Target claim: '%s'. Integration ready for API key setup.",
        config.get("project", {}).get("claim"),
    )
    return {"status": "stub_ready", "output_path": output_path or "outputs/predictions_llm.csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt LLMs for zero-shot or few-shot stance detection (STUB)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample/sample_comments.csv",
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions_llm.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zero_shot", "few_shot"],
        default="few_shot",
        help="Prompting strategy",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model identifier",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to project config.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prompt_llm(
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        prompt_mode=args.mode,
        model_provider=args.model,
    )
