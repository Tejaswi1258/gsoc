# src/preprocess.py
"""
Audio preprocessing for Spanish learner speech.

Pipeline
--------
  1. Load + resample to 16 kHz  (Whisper's required sample rate)
  2. Noise reduction             (noisereduce library — stationary mode)
  3. Silence trimming            (librosa effects)

Why each step matters for learner speech
-----------------------------------------
  Resampling   — Whisper was trained on 16 kHz audio; mismatched sample rates
                 degrade accuracy significantly.
  Noise reduction — Classroom / home recordings often contain HVAC hum, keyboard
                 clicks, or background chatter.  Removing stationary noise before
                 transcription reduces substitution errors on weak phonemes.
  Silence trim — Leading/trailing silence wastes Whisper's 30-second context
                 window and can trigger hallucinated text on empty frames.
"""

import logging

import librosa
import noisereduce as nr
import numpy as np

logger = logging.getLogger(__name__)

TARGET_SR = 16_000  # Hz — Whisper's expected sample rate


# ---------------------------------------------------------------------------
# Step 1 — Load & resample
# ---------------------------------------------------------------------------

def load_audio(file_path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Load any audio format supported by librosa/ffmpeg and resample to
    *target_sr* Hz as a mono float32 array.

    Supported formats: .wav, .mp3, .flac, .m4a (ffmpeg must be on PATH).
    """
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    logger.debug("Loaded '%s'  sr=%d  duration=%.2fs", file_path, sr, len(audio) / sr)
    return audio, sr


# ---------------------------------------------------------------------------
# Step 2 — Noise reduction
# ---------------------------------------------------------------------------

def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Remove stationary background noise using the ``noisereduce`` library.

    ``noisereduce`` implements a non-negative matrix factorisation (NMF) /
    spectral gating approach that is significantly more robust than simple
    STFT subtraction, especially for low-SNR learner recordings.

    How it works
    ------------
    1. Estimates a noise profile from the first 0.5 s of audio (assumed to
       be background noise before the speaker starts talking).
    2. Builds a spectral gate: frequency bins whose energy is below the noise
       floor are attenuated.
    3. Reconstructs the signal via inverse STFT.

    ``stationary=True`` is appropriate for constant background noise (fans,
    hum).  Set it to ``False`` for non-stationary noise (crowd, music).

    If the clip is shorter than the noise estimation window the audio is
    returned unchanged to avoid artefacts.
    """
    profile_samples = int(0.5 * sr)
    if len(audio) <= profile_samples:
        logger.debug("Clip too short for noise profile — skipping denoising")
        return audio

    noise_clip = audio[:profile_samples]
    denoised   = nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_clip,   # explicit noise reference
        stationary=True,      # constant background noise assumption
        prop_decrease=1.0,    # fully subtract estimated noise floor
    )
    logger.debug(
        "noisereduce applied  input_rms=%.4f  output_rms=%.4f",
        float(np.sqrt(np.mean(audio    ** 2))),
        float(np.sqrt(np.mean(denoised ** 2))),
    )
    return denoised.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3 — Silence trimming
# ---------------------------------------------------------------------------

def trim_silence(audio: np.ndarray, top_db: int = 20) -> np.ndarray:
    """
    Remove leading and trailing silence.

    *top_db* is the threshold (in dB below peak) below which a frame is
    considered silent.  20 dB is a good default; increase for noisier clips.
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------

def preprocess(file_path: str, denoise: bool = True) -> tuple[np.ndarray, int]:
    """
    Run the full preprocessing pipeline on a single audio file.

    Steps applied in order:
      1. Load + resample to 16 kHz
      2. Noise reduction via noisereduce  (skipped if *denoise=False*)
      3. Silence trimming

    Args:
        file_path: Path to the audio file (.wav / .mp3 / .flac / .m4a).
        denoise:   Whether to apply noise reduction (default True).

    Returns:
        ``(audio_array, sample_rate)`` ready to pass directly to Whisper.
    """
    audio, sr = load_audio(file_path)

    if denoise:
        audio = reduce_noise(audio, sr)

    audio = trim_silence(audio)

    logger.info(
        "Preprocessed '%s'  denoise=%s  final_duration=%.2fs",
        file_path, denoise, len(audio) / sr,
    )
    return audio, sr
