# src/preprocess.py
"""
Audio preprocessing for Spanish learner speech.

Pipeline:
  1. Load audio and resample to 16 kHz (Whisper requirement)
  2. Spectral noise reduction via STFT-domain subtraction
  3. Silence trimming from both ends

All functions accept/return numpy float32 arrays so they compose cleanly
with the transcription module.
"""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

TARGET_SR = 16_000  # Hz — Whisper's expected sample rate


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

def load_audio(file_path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Load an audio file and resample to *target_sr*.

    Supports .wav, .mp3, .flac, .m4a (via ffmpeg).
    Always returns a mono float32 array.
    """
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    logger.debug("Loaded '%s'  sr=%d  duration=%.2fs", file_path, sr, len(audio) / sr)
    return audio, sr


def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Spectral subtraction noise reduction.

    Estimates a noise profile from the first 0.5 s of audio (assumed to be
    background noise before the speaker starts) and subtracts it from every
    STFT frame.  This is a lightweight alternative to noisereduce/RNNoise
    that requires no extra dependencies.

    If the clip is too short to estimate a profile the audio is returned
    unchanged.
    """
    profile_len = int(0.5 * sr)
    if len(audio) <= profile_len:
        logger.debug("Audio too short for noise estimation — skipping denoising")
        return audio

    noise_profile = audio[:profile_len]

    # Full-signal STFT
    stft_full = librosa.stft(audio)
    # Noise STFT — match n_fft to full STFT's frequency bins
    n_fft = (stft_full.shape[0] - 1) * 2
    stft_noise = librosa.stft(noise_profile, n_fft=n_fft)

    # Average noise magnitude across time, keep frequency axis
    noise_mag = np.mean(np.abs(stft_noise), axis=1, keepdims=True)

    magnitude, phase = np.abs(stft_full), np.angle(stft_full)
    # Hard-floor at 0 to avoid negative magnitudes
    clean_mag = np.maximum(magnitude - noise_mag, 0.0)
    denoised = librosa.istft(clean_mag * np.exp(1j * phase))

    logger.debug("Noise reduction applied  noise_power=%.6f", float(np.mean(noise_profile ** 2)))
    return denoised


def trim_silence(audio: np.ndarray, top_db: int = 20) -> np.ndarray:
    """
    Remove leading and trailing silence.

    *top_db* controls the threshold below the peak that is considered silent.
    Lower values are more aggressive.
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------

def preprocess(file_path: str, denoise: bool = True) -> tuple[np.ndarray, int]:
    """
    Run the full preprocessing pipeline on a single audio file.

    Args:
        file_path: Path to the audio file.
        denoise:   Apply spectral noise reduction (default True).

    Returns:
        (audio_array, sample_rate) ready to pass to the transcription module.
    """
    audio, sr = load_audio(file_path)

    if denoise:
        audio = reduce_noise(audio, sr)

    audio = trim_silence(audio)

    logger.info("Preprocessed '%s'  final_duration=%.2fs", file_path, len(audio) / sr)
    return audio, sr
