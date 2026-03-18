"""
test_main.py — Unit tests for the Spanish Learner Speech Transcription Pipeline

Covers:
  - src.preprocess  : load_audio, trim_silence, reduce_noise, preprocess
  - src.postprocess : apply_phrase_corrections, apply_word_corrections, postprocess
  - src.evaluate    : compute_wer, evaluate_dataset, average_wer, analyze_errors
"""

import math
import os
import struct
import wave
import unittest

import numpy as np

from src.preprocess  import load_audio, trim_silence, reduce_noise, preprocess
from src.postprocess import (
    apply_phrase_corrections,
    apply_word_corrections,
    postprocess,
)
from src.evaluate import (
    compute_wer,
    evaluate_dataset,
    average_wer,
    analyze_errors,
)


# ---------------------------------------------------------------------------
# Shared test fixture — generates a real WAV file
# ---------------------------------------------------------------------------

def _make_wav(path: str, duration: int = 2, sr: int = 16_000, freq: float = 440.0) -> None:
    """Write a mono sine-wave WAV file to *path*."""
    n = sr * duration
    frames = struct.pack(
        "<" + "h" * n,
        *[int(32767 * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)],
    )
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(frames)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocess(unittest.TestCase):

    WAV = "_test_audio.wav"

    def setUp(self):
        _make_wav(self.WAV)

    def tearDown(self):
        if os.path.exists(self.WAV):
            os.remove(self.WAV)

    def test_load_audio_sample_rate(self):
        _, sr = load_audio(self.WAV)
        self.assertEqual(sr, 16_000)

    def test_load_audio_returns_ndarray(self):
        audio, _ = load_audio(self.WAV)
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_trim_silence_does_not_grow(self):
        import librosa
        original, sr = librosa.load(self.WAV, sr=16_000)
        trimmed = trim_silence(original)
        self.assertLessEqual(len(trimmed), len(original))

    def test_reduce_noise_returns_ndarray(self):
        import librosa
        audio, sr = librosa.load(self.WAV, sr=16_000)
        denoised = reduce_noise(audio, sr)
        self.assertIsInstance(denoised, np.ndarray)
        self.assertGreater(len(denoised), 0)

    def test_preprocess_with_denoise(self):
        audio, sr = preprocess(self.WAV, denoise=True)
        self.assertEqual(sr, 16_000)
        self.assertGreater(len(audio), 0)

    def test_preprocess_without_denoise(self):
        audio, sr = preprocess(self.WAV, denoise=False)
        self.assertEqual(sr, 16_000)
        self.assertGreater(len(audio), 0)


# ---------------------------------------------------------------------------
# Post-processing tests
# ---------------------------------------------------------------------------

class TestPostprocess(unittest.TestCase):

    def test_word_correction_nino(self):
        self.assertEqual(postprocess("nino"), "niño")

    def test_word_correction_rapido(self):
        self.assertEqual(postprocess("rapido"), "rápido")

    def test_word_correction_senor(self):
        self.assertEqual(postprocess("senor"), "señor")

    def test_phrase_correction_manzana(self):
        self.assertIn("manzana", postprocess("man zana"))

    def test_mixed_sentence(self):
        result = postprocess("el nino come una man zana")
        self.assertIn("niño", result)
        self.assertIn("manzana", result)

    def test_no_correction_needed(self):
        self.assertEqual(postprocess("hola mundo"), "hola mundo")

    def test_empty_string_passthrough(self):
        self.assertEqual(postprocess(""), "")

    def test_capitalisation_preserved(self):
        result = apply_word_corrections("Nino corre", {"nino": "niño"})
        self.assertTrue(result.startswith("Niño"))

    def test_custom_word_table(self):
        result = apply_word_corrections("gato", {"gato": "perro"})
        self.assertEqual(result, "perro")

    def test_custom_phrase_table(self):
        result = apply_phrase_corrections("hola mundo", {"hola mundo": "adiós"})
        self.assertEqual(result, "adiós")

    def test_normalization_lowercases(self):
        result = postprocess("Hola Mundo")
        self.assertEqual(result, "hola mundo")

    def test_normalization_collapses_spaces(self):
        result = postprocess("hola   mundo")
        self.assertEqual(result, "hola mundo")


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluate(unittest.TestCase):

    def test_perfect_match_wer_zero(self):
        self.assertAlmostEqual(compute_wer("hola mundo", "hola mundo"), 0.0)

    def test_full_mismatch_wer_positive(self):
        self.assertGreater(compute_wer("hola mundo", "adios amigo"), 0.0)

    def test_partial_match_wer_between_zero_and_one(self):
        score = compute_wer("hola mundo", "hola amigo")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_reference_returns_one(self):
        self.assertEqual(compute_wer("", "algo"), 1.0)

    def test_evaluate_dataset_structure(self):
        preds = {"a.wav": "hola mundo"}
        refs  = {"a.wav": "hola mundo"}
        results = evaluate_dataset(preds, refs)
        self.assertIn("a.wav", results)
        self.assertAlmostEqual(results["a.wav"]["wer"], 0.0)

    def test_evaluate_dataset_missing_reference(self):
        preds = {"b.wav": "hola"}
        refs  = {}
        results = evaluate_dataset(preds, refs)
        # No reference → WER should be 1.0
        self.assertEqual(results["b.wav"]["wer"], 1.0)

    def test_average_wer_calculation(self):
        results = {"a.wav": {"wer": 0.5}, "b.wav": {"wer": 1.0}}
        self.assertAlmostEqual(average_wer(results), 0.75)

    def test_average_wer_empty_returns_zero(self):
        self.assertEqual(average_wer({}), 0.0)

    def test_analyze_errors_keys(self):
        results = {
            "a.wav": {"reference": "hola mundo", "prediction": "hola amigo"}
        }
        summary = analyze_errors(results)
        self.assertIn("top_substitutions", summary)
        self.assertIn("top_deletions",     summary)
        self.assertIn("top_insertions",    summary)

    def test_analyze_errors_detects_substitution(self):
        results = {
            "a.wav": {"reference": "rápido", "prediction": "rapido"}
        }
        summary = analyze_errors(results)
        pairs = [ref for (ref, _), _ in summary["top_substitutions"]]
        self.assertIn("rápido", pairs)


if __name__ == "__main__":
    unittest.main()
