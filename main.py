"""
main.py — Spanish Learner Speech Transcription Pipeline
========================================================
Entry point for the GSoC speech-to-text evaluation system.

Usage:
    python main.py
    python main.py --input data/audio --refs data/references.txt
    python main.py --model small --no-denoise
    python main.py --config config.yaml
"""

import argparse
import logging
import os
import shutil
import sys

import yaml

from modules import (
    load_model,
    transcribe,
    preprocess_audio,
    postprocess_text,
    evaluate_dataset,
    average_wer,
    save_results,
    analyze_errors,
    print_error_report,
)
from modules.transcription import AUDIO_EXTENSIONS


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        logger.warning("Config file not found: %s — using defaults", config_path)
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_references(reference_file: str) -> dict[str, str]:
    """Parse pipe-delimited reference file into {filename: text} dict."""
    references = {}
    try:
        with open(reference_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 1)
                if len(parts) == 2:
                    references[parts[0].strip()] = parts[1].strip()
    except FileNotFoundError:
        logger.error("Reference file not found: %s", reference_file)
    return references


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spanish Learner Speech Transcription & Evaluation Pipeline"
    )
    parser.add_argument("--input",    default=None, help="Path to audio folder")
    parser.add_argument("--refs",     default=None, help="Path to references.txt")
    parser.add_argument("--output",   default=None, help="Path to results output file")
    parser.add_argument("--model",    default=None, help="Whisper model size (tiny/base/small/medium)")
    parser.add_argument("--config",   default="config.yaml", help="Path to config YAML")
    parser.add_argument("--no-denoise", action="store_true", help="Disable noise reduction")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    pipeline_cfg    = cfg.get("pipeline", {})
    paths_cfg       = cfg.get("paths", {})
    corrections_cfg = cfg.get("corrections", {})

    log_file = paths_cfg.get("log_file", "results/pipeline.log")
    setup_logging(log_file)

    # Resolve settings: CLI args override config, config overrides defaults
    audio_folder   = args.input  or paths_cfg.get("audio_folder",   "data")
    reference_file = args.refs   or paths_cfg.get("reference_file", "data/references.txt")
    results_file   = args.output or paths_cfg.get("results_file",   "results/predictions.txt")
    model_size     = args.model  or pipeline_cfg.get("model_size",  "base")
    language       = pipeline_cfg.get("language",    "es")
    denoise        = not args.no_denoise and pipeline_cfg.get("denoise", True)
    beam_size      = pipeline_cfg.get("beam_size",   5)
    best_of        = pipeline_cfg.get("best_of",     5)
    temperature    = pipeline_cfg.get("temperature", 0.0)

    word_corrections   = corrections_cfg.get("words")
    phrase_corrections = corrections_cfg.get("phrases")

    logger.info("=" * 55)
    logger.info("  Spanish Learner Speech Transcription Pipeline")
    logger.info("=" * 55)
    logger.info("Model: %s | Language: %s | Denoise: %s", model_size, language, denoise)

    # Preflight checks
    if shutil.which("ffmpeg") is None:
        logger.error("ffmpeg not found. Install via: choco install ffmpeg  OR  https://ffmpeg.org")
        sys.exit(1)

    references = load_references(reference_file)
    if not references:
        logger.warning("No references loaded — WER will be 1.0 for all files")

    # Load Whisper model once
    model = load_model(model_size)

    # Collect valid audio files
    audio_files = [
        f for f in sorted(os.listdir(audio_folder))
        if f.lower().endswith(AUDIO_EXTENSIONS)
        and os.path.isfile(os.path.join(audio_folder, f))
        and os.path.getsize(os.path.join(audio_folder, f)) > 0
    ]

    if not audio_files:
        logger.error("No valid audio files found in: %s", audio_folder)
        sys.exit(1)

    logger.info("Processing %d audio file(s)...\n", len(audio_files))

    predictions = {}

    for filename in audio_files:
        path = os.path.join(audio_folder, filename)
        logger.info("── %s", filename)

        try:
            audio, sr = preprocess_audio(path, denoise=denoise)
        except Exception as e:
            logger.error("Preprocessing failed for %s: %s", filename, e)
            continue

        try:
            raw_text = transcribe(
                audio, model,
                language=language,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
            )
        except Exception as e:
            logger.error("Transcription failed for %s: %s", filename, e)
            continue

        prediction = postprocess_text(
            raw_text,
            word_corrections=word_corrections,
            phrase_corrections=phrase_corrections,
        )
        predictions[filename] = prediction

        reference = references.get(filename, "")
        logger.info("  Prediction : %s", prediction)
        logger.info("  Reference  : %s", reference)

    if not predictions:
        logger.error("No files were successfully transcribed.")
        sys.exit(1)

    # Evaluation
    results  = evaluate_dataset(predictions, references)
    avg      = average_wer(results)

    logger.info("\n%s", "=" * 55)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 55)
    for filename, data in results.items():
        logger.info("  %-30s  WER: %.4f", filename, data["wer"])
    logger.info("  %-30s  WER: %.4f", "AVERAGE", avg)
    logger.info("=" * 55)

    save_results(results, results_file)

    # Error analysis
    error_summary = analyze_errors(results)
    print_error_report(error_summary)


if __name__ == "__main__":
    main()
