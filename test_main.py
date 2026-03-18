"""
test_main.py — Unit tests for the Spanish Learner Speech Transcription Pipeline
================================================================================

Run with:
    python -m pytest test_main.py -v

Coverage:
  TestPreprocess      — load_audio, trim_silence, reduce_noise, preprocess (6 tests)
  TestPostprocess     — phrase/word corrections, normalisation, edge cases  (12 tests)
  TestEvaluate        — compute_wer, evaluate_dataset, average_wer          (8 tests)
  TestErrorAnalysis   — analyze_errors, _extract_errors edge cases          (5 tests)
  TestSaveResults     — save_results file output and format                 (3 tests)
  TestMainHelpers     — _format_duration, _detect_device from main.py       (5 tests)
"""

import math
import os
import struct
import tempfile
import wave
import unittest
from unittest.mock import patch

import numpy as np

from src.evaluate import (
    analyze_errors,
    average_wer,
    compute_wer,
    evaluate_dataset,
    save_results,
)
from src.postprocess import (
    apply_phrase_corrections,
    apply_word_corrections,
    normalize,
    postprocess,
)
from src.preprocess import load_audio, preprocess, reduce_noise, trim_silence


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_wav(path: str, duration: int = 2, sr: int = 16_000, freq: float = 440.0) -> None:
    """Write a mono sine-wave WAV file to *path* for use in tests."""
    n      = sr * duration
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
        """Audio must be resampled to exactly 16 kHz for Whisper."""
        _, sr = load_audio(self.WAV)
        self.assertEqual(sr, 16_000)

    def test_load_audio_returns_float32_ndarray(self):
        """librosa should return a float32 numpy array."""
        audio, _ = load_audio(self.WAV)
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.dtype, np.float32)
        self.assertGreater(len(audio), 0)

    def test_trim_silence_does_not_grow(self):
        """Trimming can only shorten or preserve audio, never lengthen it."""
        import librosa
        original, _ = librosa.load(self.WAV, sr=16_000)
        trimmed     = trim_silence(original)
        self.assertLessEqual(len(trimmed), len(original))

    def test_reduce_noise_returns_float32_ndarray(self):
        """Denoised output must be a float32 array of positive length."""
        import librosa
        audio, sr = librosa.load(self.WAV, sr=16_000)
        denoised  = reduce_noise(audio, sr)
        self.assertIsInstance(denoised, np.ndarray)
        self.assertEqual(denoised.dtype, np.float32)
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

    # Word corrections
    def test_word_correction_nino(self):
        self.assertEqual(postprocess("nino"), "niño")

    def test_word_correction_rapido(self):
        self.assertEqual(postprocess("rapido"), "rápido")

    def test_word_correction_senor(self):
        self.assertEqual(postprocess("senor"), "señor")

    # Phrase corrections
    def test_phrase_correction_manzana(self):
        self.assertIn("manzana", postprocess("man zana"))

    def test_phrase_correction_avion(self):
        self.assertIn("avión", postprocess("a vion"))

    # Combined
    def test_mixed_sentence(self):
        result = postprocess("el nino come una man zana")
        self.assertIn("niño",    result)
        self.assertIn("manzana", result)

    # No-op cases
    def test_no_correction_needed(self):
        self.assertEqual(postprocess("hola mundo"), "hola mundo")

    def test_empty_string_passthrough(self):
        self.assertEqual(postprocess(""), "")

    def test_whitespace_only_passthrough(self):
        self.assertEqual(postprocess("   "), "   ")

    # Capitalisation
    def test_capitalisation_preserved(self):
        result = apply_word_corrections("Nino corre", {"nino": "niño"})
        self.assertTrue(result.startswith("Niño"))

    # Custom tables
    def test_custom_word_table(self):
        result = apply_word_corrections("gato", {"gato": "perro"})
        self.assertEqual(result, "perro")

    def test_custom_phrase_table(self):
        result = apply_phrase_corrections("hola mundo", {"hola mundo": "adiós"})
        self.assertEqual(result, "adiós")

    # Normalisation
    def test_normalization_lowercases(self):
        self.assertEqual(postprocess("Hola Mundo"), "hola mundo")

    def test_normalization_collapses_spaces(self):
        self.assertEqual(postprocess("hola   mundo"), "hola mundo")

    def test_normalize_strips_leading_trailing(self):
        self.assertEqual(normalize("  hola  "), "hola")

    # Partial-word matching must NOT fire
    def test_no_partial_word_match(self):
        """'rapidometro' must not be corrected — only whole-word matches."""
        result = apply_word_corrections("rapidometro", {"rapido": "rápido"})
        self.assertEqual(result, "rapidometro")


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
        results = evaluate_dataset({"a.wav": "hola mundo"}, {"a.wav": "hola mundo"})
        self.assertIn("a.wav", results)
        self.assertAlmostEqual(results["a.wav"]["wer"], 0.0)
        self.assertEqual(results["a.wav"]["prediction"], "hola mundo")
        self.assertEqual(results["a.wav"]["reference"],  "hola mundo")

    def test_evaluate_dataset_missing_reference_scores_one(self):
        results = evaluate_dataset({"b.wav": "hola"}, {})
        self.assertEqual(results["b.wav"]["wer"], 1.0)

    def test_average_wer_calculation(self):
        results = {"a.wav": {"wer": 0.5}, "b.wav": {"wer": 1.0}}
        self.assertAlmostEqual(average_wer(results), 0.75)

    def test_average_wer_empty_returns_zero(self):
        self.assertEqual(average_wer({}), 0.0)


# ---------------------------------------------------------------------------
# Error analysis tests
# ---------------------------------------------------------------------------

class TestErrorAnalysis(unittest.TestCase):

    def test_analyze_errors_returns_required_keys(self):
        results  = {"a.wav": {"reference": "hola mundo", "prediction": "hola amigo"}}
        summary  = analyze_errors(results)
        self.assertIn("top_substitutions", summary)
        self.assertIn("top_deletions",     summary)
        self.assertIn("top_insertions",    summary)

    def test_detects_substitution(self):
        results = {"a.wav": {"reference": "rápido", "prediction": "rapido"}}
        summary = analyze_errors(results)
        ref_words = [ref for (ref, _), _ in summary["top_substitutions"]]
        self.assertIn("rápido", ref_words)

    def test_detects_deletion(self):
        results = {"a.wav": {"reference": "hola mundo", "prediction": "hola"}}
        summary = analyze_errors(results)
        del_words = [w for w, _ in summary["top_deletions"]]
        self.assertIn("mundo", del_words)

    def test_detects_insertion(self):
        results = {"a.wav": {"reference": "hola", "prediction": "hola mundo"}}
        summary = analyze_errors(results)
        ins_words = [w for w, _ in summary["top_insertions"]]
        self.assertIn("mundo", ins_words)

    def test_empty_prediction_returns_empty_errors(self):
        results = {"a.wav": {"reference": "hola", "prediction": ""}}
        summary = analyze_errors(results)
        self.assertEqual(summary["top_substitutions"], [])
        self.assertEqual(summary["top_insertions"],    [])


# ---------------------------------------------------------------------------
# save_results tests
# ---------------------------------------------------------------------------

class TestSaveResults(unittest.TestCase):

    def test_creates_output_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path    = os.path.join(tmp, "out.txt")
            results = {"a.wav": {"prediction": "hola", "reference": "hola", "wer": 0.0}}
            save_results(results, path)
            self.assertTrue(os.path.exists(path))

    def test_output_contains_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path    = os.path.join(tmp, "out.txt")
            results = {"a.wav": {"prediction": "hola", "reference": "hola", "wer": 0.0}}
            save_results(results, path)
            with open(path, encoding="utf-8") as fh:
                header = fh.readline().strip()
            self.assertEqual(header, "audio_file|prediction|reference|wer")

    def test_output_contains_data_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path    = os.path.join(tmp, "out.txt")
            results = {"a.wav": {"prediction": "hola", "reference": "hola", "wer": 0.0}}
            save_results(results, path)
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
            self.assertEqual(len(lines), 2)          # header + 1 data row
            self.assertIn("a.wav", lines[1])
            self.assertIn("hola",  lines[1])


# ---------------------------------------------------------------------------
# main.py helper function tests
# ---------------------------------------------------------------------------

class TestMainHelpers(unittest.TestCase):
    """
    Tests for utility functions defined in main.py.
    Imported directly to keep tests independent of the full pipeline.
    """

    def setUp(self):
        # Import lazily so test collection doesn't trigger pipeline imports
        from main import _format_duration, _detect_device
        self._format_duration = _format_duration
        self._detect_device   = _detect_device

    def test_format_duration_seconds_only(self):
        self.assertEqual(self._format_duration(5.0),  "5.0s")
        self.assertEqual(self._format_duration(59.9), "59.9s")

    def test_format_duration_minutes_and_seconds(self):
        result = self._format_duration(90.0)
        self.assertIn("1m", result)
        self.assertIn("30.0s", result)

    def test_format_duration_zero(self):
        self.assertEqual(self._format_duration(0.0), "0.0s")

    def test_detect_device_explicit_cpu(self):
        self.assertEqual(self._detect_device("cpu"), "cpu")

    def test_detect_device_explicit_cuda(self):
        self.assertEqual(self._detect_device("cuda"), "cuda")

    def test_detect_device_auto_falls_back_to_cpu(self):
        """When no GPU is present, auto-detection must return 'cpu'."""
        with patch("torch.cuda.is_available", return_value=False):
            result = self._detect_device(None)
        self.assertEqual(result, "cpu")

    def test_detect_device_auto_picks_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            result = self._detect_device(None)
        self.assertEqual(result, "cuda")


if __name__ == "__main__":
    unittest.main()
