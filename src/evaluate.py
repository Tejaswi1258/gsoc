# src/evaluate.py
"""
Evaluation and error analysis for Spanish learner transcriptions.

Why WER?
--------
Word Error Rate (WER) is the standard metric for ASR evaluation because it
directly measures how many words the model got wrong relative to a human
transcription.  It is computed at the word level, which aligns with how
language teachers assess learner speech.

    WER = (S + D + I) / N

    S = substitutions  — wrong word predicted
    D = deletions      — reference word missed entirely
    I = insertions     — extra word hallucinated by the model
    N = total words in the reference transcript

A WER of 0.0 is a perfect transcription.
A WER of 1.0 means every reference word was wrong.
WER can exceed 1.0 when there are many insertions.

Limitations of WER for L2 speech
----------------------------------
WER treats all word errors equally, but for learner speech some errors are
more linguistically significant than others.  For example, a missing accent
("rapido" vs "rápido") is a minor orthographic error, while a completely
wrong word is a major comprehension failure.  Future work should consider
Character Error Rate (CER) or phoneme-level metrics alongside WER.

Error analysis
--------------
Beyond WER, we surface *which* words are most often substituted, deleted,
or inserted.  This helps identify systematic learner errors (e.g. missing
accents) vs. random noise, and informs which corrections to add to the
post-processing table.
"""

import logging
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

from jiwer import process_words, wer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases — makes function signatures easier to read
# ---------------------------------------------------------------------------

# Per-file result record
FileResult = Dict[str, object]   # {"prediction": str, "reference": str, "wer": float}

# Error summary returned by analyze_errors
ErrorSummary = Dict[str, List]


# ---------------------------------------------------------------------------
# WER computation
# ---------------------------------------------------------------------------

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate for a single (reference, hypothesis) pair.

    Both strings are expected to be normalised (lowercase, no extra spaces)
    before calling this function — the postprocess module handles that.

    Returns 1.0 when the reference is empty, since WER is undefined in that
    case and we want to penalise missing references in the dataset average.

    Args:
        reference:  Ground-truth transcript (human-labelled).
        hypothesis: Model prediction to evaluate.

    Returns:
        WER as a float rounded to 4 decimal places.
    """
    if not reference.strip():
        logger.warning("Empty reference string — returning WER=1.0")
        return 1.0
    return round(wer(reference, hypothesis), 4)


def evaluate_dataset(
    predictions: Dict[str, str],
    references:  Dict[str, str],
) -> Dict[str, FileResult]:
    """
    Evaluate every prediction against its reference transcript.

    Files with no matching reference are scored with WER=1.0 and an empty
    reference string, so they are included in the average but clearly marked.

    Args:
        predictions: Mapping of filename → predicted text.
        references:  Mapping of filename → reference text.

    Returns:
        Mapping of filename → {"prediction": str, "reference": str, "wer": float}
    """
    results: Dict[str, FileResult] = {}

    for filename, prediction in predictions.items():
        reference = references.get(filename, "")
        score     = compute_wer(reference, prediction)
        results[filename] = {
            "prediction": prediction,
            "reference":  reference,
            "wer":        score,
        }
        logger.info("%-30s  WER=%.4f", filename, score)

    return results


def average_wer(results: Dict[str, FileResult]) -> float:
    """
    Compute the mean WER across all evaluated files.

    Returns 0.0 for an empty results dict to avoid division-by-zero.
    """
    scores = [float(v["wer"]) for v in results.values()]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(results: Dict[str, FileResult], output_path: str) -> None:
    """
    Write per-file evaluation results to a pipe-delimited text file.

    Output format (one row per file):
        audio_file|prediction|reference|wer

    The output directory is created automatically if it does not exist.
    If *output_path* has no directory component (e.g. "predictions.txt"),
    the file is written to the current working directory.

    Args:
        results:     Output of evaluate_dataset().
        output_path: Destination file path.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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

def _extract_errors(
    reference:  str,
    hypothesis: str,
) -> Dict[str, List]:
    """
    Align reference and hypothesis at the word level using jiwer and classify
    each misalignment as a substitution, deletion, or insertion.

    Returns an empty error dict when either string is blank (no alignment
    is possible).

    Args:
        reference:  Ground-truth transcript.
        hypothesis: Model prediction.

    Returns:
        {"substitutions": [(ref_word, hyp_word), ...],
         "deletions":     [word, ...],
         "insertions":    [word, ...]}
    """
    empty: Dict[str, List] = {"substitutions": [], "deletions": [], "insertions": []}

    if not reference.strip() or not hypothesis.strip():
        return empty

    output    = process_words(reference, hypothesis)
    ref_words = output.references[0]
    hyp_words = output.hypotheses[0]

    substitutions: List[Tuple[str, str]] = []
    deletions:     List[str]             = []
    insertions:    List[str]             = []

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

    return {
        "substitutions": substitutions,
        "deletions":     deletions,
        "insertions":    insertions,
    }


def analyze_errors(
    results: Dict[str, FileResult],
    top_n:   int = 10,
) -> ErrorSummary:
    """
    Aggregate word-level errors across the whole dataset and rank them by
    frequency.

    Frequent substitutions reveal systematic learner patterns (e.g. accent
    omission).  Frequent deletions suggest words the model consistently
    mishears.  Frequent insertions suggest hallucination-prone contexts.

    Args:
        results: Output of evaluate_dataset().
        top_n:   Number of top errors to return per category.

    Returns:
        {
          "top_substitutions": [((ref_word, hyp_word), count), ...],
          "top_deletions":     [(word, count), ...],
          "top_insertions":    [(word, count), ...],
        }
    """
    subs: Counter = Counter()
    dels: Counter = Counter()
    ins:  Counter = Counter()

    for data in results.values():
        errors = _extract_errors(
            str(data.get("reference",  "")),
            str(data.get("prediction", "")),
        )
        subs.update(errors["substitutions"])
        dels.update(errors["deletions"])
        ins.update(errors["insertions"])

    logger.info(
        "Error analysis complete  substitutions=%d  deletions=%d  insertions=%d",
        len(subs), len(dels), len(ins),
    )

    return {
        "top_substitutions": subs.most_common(top_n),
        "top_deletions":     dels.most_common(top_n),
        "top_insertions":    ins.most_common(top_n),
    }


def print_error_report(summary: ErrorSummary) -> None:
    """
    Print a human-readable error analysis report to stdout.

    Intended to be called after evaluate_dataset() + analyze_errors() to
    give the researcher an immediate overview of systematic error patterns.

    Args:
        summary: Output of analyze_errors().
    """
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
