# src/transcribe.py
"""
Transcription module — wraps OpenAI Whisper for Spanish learner audio.

Key design decisions:
  - Language is *forced* to Spanish so Whisper never wastes time on language
    detection, which is unreliable for accented / non-native speech.
  - Beam search (beam_size=5) gives more accurate results than greedy decoding
    at a modest speed cost.
  - temperature=0.0 makes output deterministic and reproducible.
  - fp16=False ensures CPU compatibility (no CUDA required).
  - The model is loaded once and reused across all files to avoid the ~2 s
    reload overhead per file.
"""

import logging
import os

import numpy as np
import whisper
from tqdm import tqdm

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_size: str = "base", device: str = "cpu") -> whisper.Whisper:
    """
    Load a Whisper model by size name onto the specified device.

    Sizes (accuracy vs speed trade-off):
        tiny  < base  < small  < medium  < large
    The 'base' model is a good default for research prototypes.

    Args:
        model_size: Whisper model variant to load.
        device:     'cuda' for GPU acceleration, 'cpu' otherwise.
                    GPU inference is ~5-10x faster and enables fp16.
    """
    logger.info("Loading Whisper model: '%s' on device: '%s'", model_size, device)
    model = whisper.load_model(model_size, device=device)
    logger.info("Model loaded successfully")
    return model


# ---------------------------------------------------------------------------
# Single-file transcription
# ---------------------------------------------------------------------------

def transcribe(
    audio: "np.ndarray | str",
    model: whisper.Whisper,
    language: str = "es",
    beam_size: int = 5,
    best_of: int = 5,
    temperature: float = 0.0,
) -> str:
    """
    Transcribe one audio clip.

    Args:
        audio:      Float32 numpy array (preprocessed) *or* a file path string.
                    Passing a numpy array avoids re-loading the file from disk.
        model:      Loaded Whisper model instance.
        language:   BCP-47 language code — forced to avoid auto-detection errors.
        beam_size:  Width of beam search.  Higher = more accurate, slower.
        best_of:    Candidates sampled when temperature > 0.
        temperature: 0.0 = greedy / deterministic decoding.

    Returns:
        Stripped transcription string.
    """
    # fp16 (half-precision) is only supported on CUDA — use it automatically
    # when a GPU is available for a ~2x speed boost with no accuracy loss.
    use_fp16 = str(next(model.parameters()).device) != "cpu"

    result = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        best_of=best_of,
        temperature=temperature,
        fp16=use_fp16,
    )
    text = result["text"].strip()
    logger.debug("Transcribed: %s", text)
    return text


# ---------------------------------------------------------------------------
# Batch transcription
# ---------------------------------------------------------------------------

def batch_transcribe(
    audio_folder: str,
    model: whisper.Whisper,
    language: str = "es",
    beam_size: int = 5,
    temperature: float = 0.0,
) -> dict[str, str]:
    """
    Transcribe every audio file in *audio_folder*.

    Files are sorted alphabetically for reproducible ordering.
    A tqdm progress bar shows per-file progress in the terminal.

    Args:
        audio_folder: Directory that contains the audio files.
        model:        Loaded Whisper model.
        language:     Language code passed to every transcription call.
        beam_size:    Beam search width.
        temperature:  Decoding temperature.

    Returns:
        Dict mapping filename → raw transcription text.
        Files that fail are mapped to an empty string.
    """
    valid_files = [
        f for f in sorted(os.listdir(audio_folder))
        if f.lower().endswith(AUDIO_EXTENSIONS)
        and os.path.isfile(os.path.join(audio_folder, f))
        and os.path.getsize(os.path.join(audio_folder, f)) > 0
    ]

    if not valid_files:
        logger.warning("No valid audio files found in '%s'", audio_folder)
        return {}

    logger.info("Batch transcribing %d file(s) from '%s'", len(valid_files), audio_folder)

    results: dict[str, str] = {}
    for filename in tqdm(valid_files, desc="Transcribing", unit="file"):
        path = os.path.join(audio_folder, filename)
        try:
            results[filename] = transcribe(
                path, model,
                language=language,
                beam_size=beam_size,
                temperature=temperature,
            )
        except Exception as exc:
            logger.error("Failed to transcribe '%s': %s", filename, exc)
            results[filename] = ""

    return results
