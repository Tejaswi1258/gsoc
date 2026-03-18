# src/evaluate.py
"""
Evaluation and error analysis for Spanish learner transcriptions.

Metrics
-------
Word Error Rate (WER) — the standard metric for ASR evaluation.

    WER = (S + D + I) / N

    S = substitutions  (wrong word)
    D = deletions      (missed word)
    I = insertions     (extra word)
    N = total words in reference

A WER of 0.0 is perfect; 1.0 means every reference word was wrong.

Error analysis
--------------
Beyond WER, we surface *which* words are most often substituted, deleted,
or inserted.  This helps identify systematic learner errors (e.g. missing
accents) vs. random noise.
"""

import logging
import os
from collections import Counter

from jiwer import process_words, wer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WER helpers
# ---------------------------------------------------------------------------

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute WER for a single (reference, hypothesis) pair.

    Returns 1.0 when the reference is empty (WER is undefined in that case).
    """
    if not reference.strip():
        logger.warning("Empty reference string — returning WER=1.0")
        return 1.0
    return round(wer(reference, hypothesis), 4)


def evaluate_dataset(
    predictions: dict[str, str],
    references: dict[str, str],
) -> dict[str, dict]:
    """
    Evaluate every prediction against its reference.

    Args:
        predictions: {filename: predicted_text}
        references:  {filename: reference_text}

    Returns:
        {filename: {"prediction": str, "reference": str, "wer": float}}
    """
    results: dict[str, dict] = {}
    for filename, prediction in predictions.items():
        reference = references.get(filename, "")
        score = compute_wer(reference, prediction)
        results[filename] = {
            "prediction": prediction,
            "reference":  reference,
            "wer":        score,
        }
        logger.info("%-30s  WER=%.4f", filename, score)
    return results


def average_wer(results: dict[str, dict]) -> float:
    """Return the mean WER across all evaluated files."""
    scores = [v["wer"] for v in results.values()]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(results: dict[str, dict], output_path: str) -> None:
    """
    Write per-file results to a pipe-delimited text file.

    Header:  audio_file | prediction | reference | wer
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("audio_file|prediction|reference|wer\n")
        for filename, data in results.items():
            fh.write(
                f"{filename}|{data['prediction']}|{data['reference']}|{data['wer']}\n"
            )
    logger.info("Results saved → '%s'", output_path)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def _extract_errors(reference: str, hypothesis: str) -> dict:
    """
    Align reference and hypothesis at the word level and classify each
    misalignment as a substitution, deletion, or insertion.
    """
    if not reference.strip() or not hypothesis.strip():
        return {"substitutions": [], "deletions": [], "insertions": []}

    output    = process_words(reference, hypothesis)
    ref_words = output.references[0]
    hyp_words = output.hypotheses[0]

    substitutions: list[tuple[str, str]] = []
    deletions:     list[str]             = []
    insertions:    list[str]             = []

    for chunk in output.alignments[0]:
        if chunk.type == "substitute":
            for r, h in zip(
                ref_words[chunk.ref_start_idx:chunk.ref_end_idx],
                hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx],
            ):
                substitutions.append((r, h))
        elif chunk.type == "delete":
            deletions.extend(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
        elif chunk.type == "insert":
            insertions.extend(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])

    return {"substitutions": substitutions, "deletions": deletions, "insertions": insertions}


def analyze_errors(results: dict[str, dict], top_n: int = 10) -> dict:
    """
    Aggregate word-level errors across the whole dataset.

    Args:
        results: Output of evaluate_dataset().
        top_n:   How many top errors to return per category.

    Returns:
        {
          "top_substitutions": [((ref, hyp), count), ...],
          "top_deletions":     [(word, count), ...],
          "top_insertions":    [(word, count), ...],
        }
    """
    subs = Counter()
    dels = Counter()
    ins  = Counter()

    for data in results.values():
        errors = _extract_errors(data["reference"], data["prediction"])
        subs.update(errors["substitutions"])
        dels.update(errors["deletions"])
        ins.update(errors["insertions"])

    logger.info(
        "Error analysis done  substitutions=%d  deletions=%d  insertions=%d",
        len(subs), len(dels), len(ins),
    )
    return {
        "top_substitutions": subs.most_common(top_n),
        "top_deletions":     dels.most_common(top_n),
        "top_insertions":    ins.most_common(top_n),
    }


def print_error_report(summary: dict) -> None:
    """Print a formatted error analysis report to stdout."""
    sep = "=" * 55
    print(f"\n{sep}")
    print("  ERROR ANALYSIS REPORT")
    print(sep)

    print("\n[Substitutions]  reference → hypothesis")
    if summary["top_substitutions"]:
        for (ref, hyp), n in summary["top_substitutions"]:
            print(f"  '{ref}' → '{hyp}'  (×{n})")
    else:
        print("  None found.")

    print("\n[Deletions]  words missed by the model")
    if summary["top_deletions"]:
        for word, n in summary["top_deletions"]:
            print(f"  '{word}'  (×{n})")
    else:
        print("  None found.")

    print("\n[Insertions]  extra words added by the model")
    if summary["top_insertions"]:
        for word, n in summary["top_insertions"]:
            print(f"  '{word}'  (×{n})")
    else:
        print("  None found.")

    print(f"{sep}\n")
