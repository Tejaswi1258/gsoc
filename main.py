"""
main.py — Spanish Learner Speech Transcription Pipeline
========================================================

Orchestrates the full pipeline:
  preprocess → transcribe → postprocess → evaluate → error analysis

Usage examples
--------------
  # Run with config.yaml defaults
  python main.py

  # Custom paths
  python main.py --input data/audio --refs data/references.txt --output results/predictions.txt

  # Use a more accurate model, skip noise reduction
  python main.py --model small --no-denoise

  # Force CPU even if a GPU is available
  python main.py --device cpu
"""

import argparse
import logging
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm

from src.evaluate    import analyze_errors, average_wer, evaluate_dataset, print_error_report, save_results
from src.postprocess import DEFAULT_PHRASE_CORRECTIONS, DEFAULT_WORD_CORRECTIONS, postprocess
from src.preprocess  import preprocess
from src.transcribe  import AUDIO_EXTENSIONS, load_model, transcribe


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_file: str) -> None:
    """
    Configure logging to write to both stdout and a log file.
    The log file is created (along with any missing parent dirs) automatically.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    """Load YAML config file. Returns empty dict if the file doesn't exist."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_references(ref_file: str) -> Dict[str, str]:
    """
    Parse a pipe-delimited reference file into {filename: reference_text}.

    Expected format (one entry per line):
        audio1.wav|el niño come una manzana
        audio2.wav|la mujer bebe agua

    Lines that are blank or malformed are silently skipped.
    """
    refs: Dict[str, str] = {}
    try:
        with open(ref_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 1)
                if len(parts) == 2:
                    refs[parts[0].strip()] = parts[1].strip()
    except FileNotFoundError:
        logger.error("Reference file not found: '%s'", ref_file)
    return refs


def _collect_audio_files(folder: str) -> List[str]:
    """
    Return a sorted list of non-empty audio filenames inside *folder*.
    Only files with recognised extensions are included.
    """
    return [
        f for f in sorted(os.listdir(folder))
        if f.lower().endswith(AUDIO_EXTENSIONS)
        and os.path.isfile(os.path.join(folder, f))
        and os.path.getsize(os.path.join(folder, f)) > 0
    ]


def _detect_device(requested: Optional[str]) -> str:
    """
    Resolve the compute device to use for Whisper.

    Priority:
      1. Explicit --device CLI flag
      2. CUDA if a GPU is available
      3. CPU fallback
    """
    if requested:
        return requested
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _format_duration(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable string, e.g. '1m 23.4s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs    = seconds % 60
    return f"{minutes}m {secs:.1f}s"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Spanish Learner Speech Transcription & Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      default=None, metavar="DIR",    help="Folder containing audio files")
    p.add_argument("--refs",       default=None, metavar="FILE",   help="Pipe-delimited reference file")
    p.add_argument("--output",     default=None, metavar="FILE",   help="Results output file")
    p.add_argument("--model",      default=None, metavar="SIZE",   help="Whisper model size (tiny/base/small/medium/large)")
    p.add_argument("--device",     default=None, metavar="DEVICE", help="Compute device: cuda | cpu (auto-detected if omitted)")
    p.add_argument("--config",     default="config.yaml",          help="Path to YAML config file")
    p.add_argument("--no-denoise", action="store_true",            help="Disable spectral noise reduction")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _process_file(
    filename:           str,
    audio_folder:       str,
    model:              object,
    references:         Dict[str, str],
    denoise:            bool,
    language:           str,
    beam_size:          int,
    temperature:        float,
    word_corrections:   Dict[str, str],
    phrase_corrections: Dict[str, str],
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Run the full pipeline for a single audio file.

    Returns:
        (prediction, reference, elapsed_seconds)
        prediction is None if any pipeline step failed.
    """
    path      = os.path.join(audio_folder, filename)
    reference = references.get(filename)          # None if no reference exists
    t_start   = time.perf_counter()

    # Warn early so the user knows WER won't be meaningful for this file
    if reference is None:
        logger.warning("  No reference found for '%s' — WER will be 1.0", filename)

    # Step 1 — Preprocess
    try:
        audio, _sr = preprocess(path, denoise=denoise)
    except Exception as exc:
        logger.error("  Preprocessing failed: %s", exc)
        return None, reference, time.perf_counter() - t_start

    # Step 2 — Transcribe
    try:
        raw_text = transcribe(
            audio, model,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
        )
    except Exception as exc:
        logger.error("  Transcription failed: %s", exc)
        return None, reference, time.perf_counter() - t_start

    # Step 3 — Post-process
    prediction = postprocess(
        raw_text,
        word_corrections=word_corrections,
        phrase_corrections=phrase_corrections,
    )

    elapsed = time.perf_counter() - t_start
    logger.info("  Prediction : %s", prediction)
    logger.info("  Reference  : %s", reference or "(none)")
    logger.info("  Time       : %s", _format_duration(elapsed))

    return prediction, reference, elapsed


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    pipeline_start = time.perf_counter()

    args = _parse_args()
    cfg  = _load_config(args.config)

    pipe_cfg = cfg.get("pipeline",    {})
    path_cfg = cfg.get("paths",       {})
    corr_cfg = cfg.get("corrections", {})

    # Resolve settings: CLI flag > config value > hard-coded default
    audio_folder   = args.input  or path_cfg.get("audio_folder",   "data/audio")
    reference_file = args.refs   or path_cfg.get("reference_file", "data/references.txt")
    results_file   = args.output or path_cfg.get("results_file",   "results/predictions.txt")
    log_file       = path_cfg.get("log_file", "results/pipeline.log")
    model_size     = args.model  or pipe_cfg.get("model_size",  "base")
    language       = pipe_cfg.get("language",    "es")
    denoise        = (not args.no_denoise) and pipe_cfg.get("denoise", True)
    beam_size      = pipe_cfg.get("beam_size",   5)
    temperature    = pipe_cfg.get("temperature", 0.0)
    device         = _detect_device(args.device)

    word_corrections:   Dict[str, str] = corr_cfg.get("words")   or DEFAULT_WORD_CORRECTIONS
    phrase_corrections: Dict[str, str] = corr_cfg.get("phrases") or DEFAULT_PHRASE_CORRECTIONS

    _setup_logging(log_file)

    # ── Banner ──────────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("  Spanish Learner Speech Transcription Pipeline")
    logger.info("=" * 55)
    logger.info(
        "model=%-8s  language=%s  device=%-4s  denoise=%s  beam=%d",
        model_size, language, device, denoise, beam_size,
    )

    # ── Preflight checks ────────────────────────────────────────────────
    if shutil.which("ffmpeg") is None:
        logger.error(
            "ffmpeg not found on PATH.\n"
            "  Windows : choco install ffmpeg\n"
            "  macOS   : brew install ffmpeg\n"
            "  Ubuntu  : sudo apt install ffmpeg"
        )
        sys.exit(1)

    if device == "cuda":
        logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("Running on CPU — transcription will be slower")

    # ── Load references ─────────────────────────────────────────────────
    references: Dict[str, str] = _load_references(reference_file)
    if not references:
        logger.warning("No references loaded — WER will be 1.0 for all files")

    # ── Collect audio files ─────────────────────────────────────────────
    audio_files: List[str] = _collect_audio_files(audio_folder)
    if not audio_files:
        logger.error("No valid audio files found in '%s'", audio_folder)
        sys.exit(1)

    logger.info("Found %d audio file(s) to process\n", len(audio_files))

    # ── Load Whisper model once ──────────────────────────────────────────
    model = load_model(model_size, device=device)

    # ── Per-file pipeline with progress bar ─────────────────────────────
    predictions:  Dict[str, str]   = {}   # filename → clean transcription
    file_times:   Dict[str, float] = {}   # filename → elapsed seconds
    failed_files: List[str]        = []   # files that could not be processed

    progress = tqdm(audio_files, desc="Processing", unit="file", ncols=70)

    for filename in progress:
        progress.set_postfix_str(filename)
        logger.info("── [%d/%d] %s", audio_files.index(filename) + 1, len(audio_files), filename)

        prediction, reference, elapsed = _process_file(
            filename, audio_folder, model, references,
            denoise, language, beam_size, temperature,
            word_corrections, phrase_corrections,
        )

        file_times[filename] = elapsed

        if prediction is None:
            failed_files.append(filename)
        else:
            predictions[filename] = prediction

    progress.close()

    # ── Failure summary ──────────────────────────────────────────────────
    if failed_files:
        logger.warning("\n%d file(s) failed to process:", len(failed_files))
        for f in failed_files:
            logger.warning("  ✗ %s", f)

    if not predictions:
        logger.error("No files were successfully transcribed. Exiting.")
        sys.exit(1)

    # ── Evaluation ───────────────────────────────────────────────────────
    results = evaluate_dataset(predictions, references)
    avg     = average_wer(results)

    logger.info("\n%s", "=" * 55)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 55)
    for fname, data in results.items():
        has_ref = bool(data.get("reference"))
        wer_str = f"WER: {data['wer']:.4f}" if has_ref else "WER: n/a (no reference)"
        logger.info("  %-32s  %s  [%.1fs]", fname, wer_str, file_times.get(fname, 0.0))
    logger.info("  %-32s  WER: %.4f", "AVERAGE", avg)
    logger.info("=" * 55)

    save_results(results, results_file)

    # ── Error analysis ───────────────────────────────────────────────────
    print_error_report(analyze_errors(results))

    # ── Total execution time ─────────────────────────────────────────────
    total_elapsed = time.perf_counter() - pipeline_start
    logger.info("Total pipeline time: %s", _format_duration(total_elapsed))
    logger.info("Results saved to: %s", results_file)
    logger.info("Log saved to:     %s", log_file)


if __name__ == "__main__":
    main()
