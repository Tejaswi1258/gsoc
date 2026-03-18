# modules/evaluation.py
"""
Evaluation module for speech transcription quality.
Computes Word Error Rate (WER) per file and dataset-level averages.
"""

import os
import logging
from jiwer import wer

logger = logging.getLogger(__name__)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate between reference and hypothesis.

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference words.

    Returns 1.0 if reference is empty (undefined WER).
    """
    if not reference.strip():
        logger.warning("Empty reference — returning WER=1.0")
        return 1.0
    return round(wer(reference, hypothesis), 4)


def evaluate_dataset(
    predictions: dict[str, str],
    references: dict[str, str],
) -> dict[str, dict]:
    """
    Evaluate all predictions against references.

    Args:
        predictions: Dict of filename -> predicted text
        references:  Dict of filename -> reference text

    Returns:
        Dict of filename -> {"prediction", "reference", "wer"}
    """
    results = {}
    for filename, prediction in predictions.items():
        reference = references.get(filename, "")
        score = compute_wer(reference, prediction)
        results[filename] = {
            "prediction": prediction,
            "reference":  reference,
            "wer":        score,
        }
        logger.info("%s | WER=%.4f", filename, score)
    return results


def average_wer(results: dict[str, dict]) -> float:
    """Compute mean WER across all evaluated files."""
    scores = [v["wer"] for v in results.values()]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def save_results(results: dict[str, dict], output_path: str) -> None:
    """
    Save per-file results to a pipe-delimited text file.

    Format: audio_file | prediction | reference | WER
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("audio_file|prediction|reference|wer\n")
        for filename, data in results.items():
            f.write(f"{filename}|{data['prediction']}|{data['reference']}|{data['wer']}\n")
    logger.info("Results saved to %s", output_path)
