import os
import math
import struct
import wave
import unittest
import numpy as np

from modules.preprocessing import preprocess_audio, trim_silence, reduce_noise
from modules.postprocessing import postprocess_text, apply_phrase_corrections, apply_word_corrections
from modules.evaluation import compute_wer, evaluate_dataset, average_wer


def create_test_wav(path, duration=2, sample_rate=16000, frequency=440.0):
    num_samples = sample_rate * duration
    frames = struct.pack(
        '<' + 'h' * num_samples,
        *[int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate)) for i in range(num_samples)]
    )
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessAudio(unittest.TestCase):

    def setUp(self):
        self.wav = "test_temp.wav"
        create_test_wav(self.wav)

    def tearDown(self):
        if os.path.exists(self.wav):
            os.remove(self.wav)

    def test_returns_correct_sample_rate(self):
        _, sr = preprocess_audio(self.wav)
        self.assertEqual(sr, 16000)

    def test_returns_numpy_array(self):
        audio, _ = preprocess_audio(self.wav)
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_silence_trimmed(self):
        import librosa
        original, sr = librosa.load(self.wav, sr=16000)
        trimmed = trim_silence(original)
        self.assertLessEqual(len(trimmed), len(original))

    def test_denoise_preserves_shape(self):
        import librosa
        audio, sr = librosa.load(self.wav, sr=16000)
        denoised = reduce_noise(audio, sr)
        self.assertIsInstance(denoised, np.ndarray)
        self.assertGreater(len(denoised), 0)

    def test_preprocess_without_denoise(self):
        audio, sr = preprocess_audio(self.wav, denoise=False)
        self.assertEqual(sr, 16000)
        self.assertGreater(len(audio), 0)


# ---------------------------------------------------------------------------
# Postprocessing tests
# ---------------------------------------------------------------------------

class TestPostprocessText(unittest.TestCase):

    def test_word_correction(self):
        self.assertEqual(postprocess_text("nino"), "niño")
        self.assertEqual(postprocess_text("rapido"), "rápido")
        self.assertEqual(postprocess_text("senor"), "señor")

    def test_phrase_correction(self):
        result = postprocess_text("man zana")
        self.assertEqual(result, "manzana")

    def test_mixed_corrections(self):
        result = postprocess_text("el nino come una man zana")
        self.assertIn("niño", result)
        self.assertIn("manzana", result)

    def test_no_correction_needed(self):
        self.assertEqual(postprocess_text("hola mundo"), "hola mundo")

    def test_empty_string(self):
        self.assertEqual(postprocess_text(""), "")

    def test_custom_corrections(self):
        result = apply_word_corrections("gato", {"gato": "perro"})
        self.assertEqual(result, "perro")

    def test_capitalisation_preserved(self):
        result = apply_word_corrections("Nino", {"nino": "niño"})
        self.assertEqual(result, "Niño")


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluation(unittest.TestCase):

    def test_perfect_match(self):
        self.assertAlmostEqual(compute_wer("hola mundo", "hola mundo"), 0.0)

    def test_full_mismatch(self):
        self.assertGreater(compute_wer("hola mundo", "adios amigo"), 0.0)

    def test_partial_match(self):
        score = compute_wer("hola mundo", "hola amigo")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_reference_returns_one(self):
        self.assertEqual(compute_wer("", "algo"), 1.0)

    def test_evaluate_dataset(self):
        predictions = {"a.wav": "hola mundo"}
        references  = {"a.wav": "hola mundo"}
        results = evaluate_dataset(predictions, references)
        self.assertAlmostEqual(results["a.wav"]["wer"], 0.0)

    def test_average_wer(self):
        results = {
            "a.wav": {"wer": 0.5},
            "b.wav": {"wer": 1.0},
        }
        self.assertAlmostEqual(average_wer(results), 0.75)

    def test_average_wer_empty(self):
        self.assertEqual(average_wer({}), 0.0)


if __name__ == "__main__":
    unittest.main()
