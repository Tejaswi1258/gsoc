# modules/transcription.py
"""
Transcription module using OpenAI Whisper.
Supports single-file and batch transcription with optimized decoding parameters.
"""

import logging
import os
import numpy as np
import whisper

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.m4a')


def load_model(model_size: str = "base") -> whisper.Whisper:
    """Load Whisper model. Use 'small' or 'medium' for better accuracy."""
    logger.info("Loading Whisper model: %s", model_size)
    return whisper.load_model(model_size)


def transcribe(
    audio: np.ndarray | str,
    model: whisper.Whisper,
    language: str = "es",
    beam_size: int = 5,
    best_of: int = 5,
    temperature: float = 0.0,
) -> str:
    """
    Transcribe a single audio array or file path using Whisper.

    Args:
        audio: Preprocessed numpy array (float32) or path to audio file
        model: Loaded Whisper model instance
        language: Force language (default: Spanish)
        beam_size: Beam search width — higher = more accurate but slower
        best_of: Number of candidates when using sampling
        temperature: 0.0 = greedy decoding (most deterministic)

    Returns:
        Transcribed text string
    """
    result = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        best_of=best_of,
        temperature=temperature,
        fp16=False,  # CPU-safe
    )
    text = result["text"].strip()
    logger.debug("Transcription result: %s", text)
    return text


def batch_transcribe(
    audio_folder: str,
    model: whisper.Whisper,
    language: str = "es",
) -> dict[str, str]:
    """
    Transcribe all audio files in a folder.

    Args:
        audio_folder: Directory containing audio files
        model: Loaded Whisper model instance
        language: Language code for transcription

    Returns:
        Dict mapping filename -> raw transcription text
    """
    results = {}

    audio_files = [
        f for f in sorted(os.listdir(audio_folder))
        if f.lower().endswith(AUDIO_EXTENSIONS)
        and os.path.isfile(os.path.join(audio_folder, f))
        and os.path.getsize(os.path.join(audio_folder, f)) > 0
    ]

    if not audio_files:
        logger.warning("No valid audio files found in: %s", audio_folder)
        return results

    logger.info("Found %d audio file(s) to transcribe", len(audio_files))

    for filename in audio_files:
        path = os.path.join(audio_folder, filename)
        logger.info("Transcribing: %s", filename)
        try:
            results[filename] = transcribe(path, model, language=language)
        except Exception as e:
            logger.error("Failed to transcribe %s: %s", filename, e)
            results[filename] = ""

    return results
