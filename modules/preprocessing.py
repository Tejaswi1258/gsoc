# modules/preprocessing.py
"""
Audio preprocessing pipeline for Spanish learner speech.
Handles loading, resampling, noise reduction, and silence trimming.
"""

import logging
import numpy as np
import librosa

logger = logging.getLogger(__name__)

TARGET_SR = 16000  # Whisper expects 16kHz


def load_audio(file_path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load and resample audio to target sample rate."""
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    logger.debug("Loaded %s | sr=%d | duration=%.2fs", file_path, sr, len(audio) / sr)
    return audio, sr


def trim_silence(audio: np.ndarray, top_db: int = 20) -> np.ndarray:
    """Remove leading and trailing silence."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Simple spectral noise reduction using a noise profile
    estimated from the first 0.5s of audio (assumed to be background noise).
    """
    noise_sample_len = int(0.5 * sr)
    if len(audio) <= noise_sample_len:
        return audio

    # Estimate noise profile from first 0.5s
    noise_profile = audio[:noise_sample_len]
    noise_power = np.mean(noise_profile ** 2)

    # Spectral subtraction in STFT domain
    stft = librosa.stft(audio)
    noise_stft = librosa.stft(noise_profile, n_fft=stft.shape[0] * 2 - 2)
    noise_mag = np.mean(np.abs(noise_stft), axis=1, keepdims=True)

    mag, phase = np.abs(stft), np.angle(stft)
    mag_denoised = np.maximum(mag - noise_mag, 0.0)
    denoised = librosa.istft(mag_denoised * np.exp(1j * phase))

    logger.debug("Noise reduction applied | noise_power=%.6f", noise_power)
    return denoised


def preprocess_audio(file_path: str, denoise: bool = True) -> tuple[np.ndarray, int]:
    """
    Full preprocessing pipeline:
      1. Load + resample to 16kHz
      2. Noise reduction (optional)
      3. Silence trimming

    Args:
        file_path: Path to audio file (.wav, .mp3, .flac, .m4a)
        denoise: Whether to apply spectral noise reduction

    Returns:
        Tuple of (processed audio array, sample rate)
    """
    audio, sr = load_audio(file_path)

    if denoise:
        audio = reduce_noise(audio, sr)

    audio = trim_silence(audio)

    logger.info("Preprocessed %s | final duration=%.2fs", file_path, len(audio) / sr)
    return audio, sr
