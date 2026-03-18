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
  python main.py --input data --refs data/references.txt --output results/predictions.txt

  # Use a more accurate model, skip noise reduction
  python main.py --model small --no-denoise

  # Point to a different config file
  python main.py --config my_config.yaml
"""

import argparse
import logging
import os
import shutil
import sys

import yaml

from src.preprocess  import preprocess
from src.transcribe  import load_model, transcribe, AUDIO_EXTENSIONS
from src.postprocess import postprocess, DEFAULT_WORD_CORRECTIONS, DEFAULT_PHRASE_CORRECTIONS
from src.evaluate    import evaluate_dataset, average_wer, save_results, analyze_errors, print_error_report


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_file: str) -> None:
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
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_references(ref_file: str) -> dict[str, str]:
    """Parse  filename|reference text  lines into a dict."""
    refs: dict[str, str] = {}
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


def _collect_audio_files(folder: str) -> list[str]:
    return [
        f for f in sorted(os.listdir(folder))
        if f.lower().endswith(AUDIO_EXTENSIONS)
        and os.path.isfile(os.path.join(folder, f))
        and os.path.getsize(os.path.join(folder, f)) > 0
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Spanish Learner Speech Transcription & Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      default=None, metavar="DIR",  help="Folder containing audio files")
    p.add_argument("--refs",       default=None, metavar="FILE", help="Pipe-delimited reference file")
    p.add_argument("--output",     default=None, metavar="FILE", help="Results output file")
    p.add_argument("--model",      default=None, metavar="SIZE", help="Whisper model size (tiny/base/small/medium/large)")
    p.add_argument("--config",     default="config.yaml",        help="Path to YAML config file")
    p.add_argument("--no-denoise", action="store_true",          help="Disable spectral noise reduction")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    cfg  = _load_config(args.config)

    pipe_cfg  = cfg.get("pipeline",     {})
    path_cfg  = cfg.get("paths",        {})
    corr_cfg  = cfg.get("corrections",  {})

    # Resolve settings: CLI > config > hard-coded defaults
    audio_folder   = args.input  or path_cfg.get("audio_folder",   "data")
    reference_file = args.refs   or path_cfg.get("reference_file", "data/references.txt")
    results_file   = args.output or path_cfg.get("results_file",   "results/predictions.txt")
    log_file       = path_cfg.get("log_file", "results/pipeline.log")
    model_size     = args.model  or pipe_cfg.get("model_size",  "base")
    language       = pipe_cfg.get("language",    "es")
    denoise        = (not args.no_denoise) and pipe_cfg.get("denoise", True)
    beam_size      = pipe_cfg.get("beam_size",   5)
    temperature    = pipe_cfg.get("temperature", 0.0)

    word_corrections   = corr_cfg.get("words")   or DEFAULT_WORD_CORRECTIONS
    phrase_corrections = corr_cfg.get("phrases") or DEFAULT_PHRASE_CORRECTIONS

    _setup_logging(log_file)

    # ------------------------------------------------------------------
    logger.info("=" * 55)
    logger.info("  Spanish Learner Speech Transcription Pipeline")
    logger.info("=" * 55)
    logger.info("model=%s  language=%s  denoise=%s  beam=%d", model_size, language, denoise, beam_size)

    # Preflight: ffmpeg must be on PATH (Whisper uses it for audio decoding)
    if shutil.which("ffmpeg") is None:
        logger.error(
            "ffmpeg not found on PATH.\n"
            "  Windows : choco install ffmpeg\n"
            "  macOS   : brew install ffmpeg\n"
            "  Ubuntu  : sudo apt install ffmpeg"
        )
        sys.exit(1)

    # Load references
    references = _load_references(reference_file)
    if not references:
        logger.warning("No references loaded — WER will be 1.0 for all files")

    # Collect audio files
    audio_files = _collect_audio_files(audio_folder)
    if not audio_files:
        logger.error("No valid audio files found in '%s'", audio_folder)
        sys.exit(1)

    logger.info("Found %d audio file(s) to process", len(audio_files))

    # Load Whisper model once
    model = load_model(model_size)

    # ------------------------------------------------------------------
    # Per-file pipeline
    # ------------------------------------------------------------------
    predictions: dict[str, str] = {}

    for filename in audio_files:
        path = os.path.join(audio_folder, filename)
        logger.info("── %s", filename)

        # 1. Preprocess
        try:
            audio, _sr = preprocess(path, denoise=denoise)
        except Exception as exc:
            logger.error("Preprocessing failed for '%s': %s", filename, exc)
            continue

        # 2. Transcribe
        try:
            raw_text = transcribe(
                audio, model,
                language=language,
                beam_size=beam_size,
                temperature=temperature,
            )
        except Exception as exc:
            logger.error("Transcription failed for '%s': %s", filename, exc)
            continue

        # 3. Post-process
        clean_text = postprocess(
            raw_text,
            word_corrections=word_corrections,
            phrase_corrections=phrase_corrections,
        )
        predictions[filename] = clean_text

        logger.info("  Prediction : %s", clean_text)
        logger.info("  Reference  : %s", references.get(filename, "(no reference)"))

    if not predictions:
        logger.error("No files were successfully transcribed.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    results = evaluate_dataset(predictions, references)
    avg     = average_wer(results)

    logger.info("\n%s", "=" * 55)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 55)
    for fname, data in results.items():
        logger.info("  %-32s  WER: %.4f", fname, data["wer"])
    logger.info("  %-32s  WER: %.4f", "AVERAGE", avg)
    logger.info("=" * 55)

    save_results(results, results_file)

    # ------------------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------------------
    print_error_report(analyze_errors(results))


if __name__ == "__main__":
    main()
