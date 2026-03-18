# modules/error_analysis.py
"""
Error analysis module for transcription evaluation.
Identifies frequent substitution, deletion, and insertion errors.
"""

import logging
from collections import Counter
from jiwer import process_words

logger = logging.getLogger(__name__)


def extract_errors(reference: str, hypothesis: str) -> dict:
    """
    Extract word-level substitutions, deletions, and insertions
    between a reference and hypothesis string.

    Returns:
        Dict with keys: substitutions, deletions, insertions
    """
    if not reference.strip() or not hypothesis.strip():
        return {"substitutions": [], "deletions": [], "insertions": []}

    output = process_words(reference, hypothesis)
    ref_words = output.references[0]
    hyp_words = output.hypotheses[0]

    substitutions, deletions, insertions = [], [], []

    ref_idx = hyp_idx = 0
    for chunk in output.alignments[0]:
        if chunk.type == "equal":
            ref_idx += chunk.ref_end_idx - chunk.ref_start_idx
            hyp_idx += chunk.hyp_end_idx - chunk.hyp_start_idx
        elif chunk.type == "substitute":
            for r, h in zip(
                ref_words[chunk.ref_start_idx:chunk.ref_end_idx],
                hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx],
            ):
                substitutions.append((r, h))
        elif chunk.type == "delete":
            deletions.extend(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
        elif chunk.type == "insert":
            insertions.extend(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])

    return {
        "substitutions": substitutions,
        "deletions":     deletions,
        "insertions":    insertions,
    }


def analyze_errors(results: dict[str, dict], top_n: int = 10) -> dict:
    """
    Aggregate errors across all files and return a summary.

    Args:
        results: Output from evaluation.evaluate_dataset()
        top_n:   Number of top errors to surface per category

    Returns:
        Summary dict with top substitutions, deletions, insertions
    """
    all_substitutions = Counter()
    all_deletions     = Counter()
    all_insertions    = Counter()

    for data in results.values():
        errors = extract_errors(data["reference"], data["prediction"])
        all_substitutions.update(errors["substitutions"])
        all_deletions.update(errors["deletions"])
        all_insertions.update(errors["insertions"])

    summary = {
        "top_substitutions": all_substitutions.most_common(top_n),
        "top_deletions":     all_deletions.most_common(top_n),
        "top_insertions":    all_insertions.most_common(top_n),
    }

    logger.info("Error analysis complete | substitutions=%d unique pairs", len(all_substitutions))
    return summary


def print_error_report(summary: dict) -> None:
    """Print a human-readable error analysis report to stdout."""
    print("\n" + "=" * 55)
    print("  ERROR ANALYSIS REPORT")
    print("=" * 55)

    print("\n[Substitutions] (reference → hypothesis)")
    if summary["top_substitutions"]:
        for (ref, hyp), count in summary["top_substitutions"]:
            print(f"  '{ref}' → '{hyp}'  (x{count})")
    else:
        print("  None found.")

    print("\n[Deletions] (words missed by model)")
    if summary["top_deletions"]:
        for word, count in summary["top_deletions"]:
            print(f"  '{word}'  (x{count})")
    else:
        print("  None found.")

    print("\n[Insertions] (extra words added by model)")
    if summary["top_insertions"]:
        for word, count in summary["top_insertions"]:
            print(f"  '{word}'  (x{count})")
    else:
        print("  None found.")

    print("=" * 55 + "\n")
