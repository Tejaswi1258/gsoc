import os
import wave
import struct
import math
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from main import preprocess_audio, postprocess_text, evaluate


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


class TestPreprocessAudio(unittest.TestCase):

    def setUp(self):
        self.test_wav = "test_temp.wav"
        create_test_wav(self.test_wav)

    def tearDown(self):
        if os.path.exists(self.test_wav):
            os.remove(self.test_wav)

    def test_returns_audio_and_sr(self):
        audio, sr = preprocess_audio(self.test_wav)
        self.assertEqual(sr, 16000)
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_silence_trimmed(self):
        audio, sr = preprocess_audio(self.test_wav)
        # trimmed audio should be <= original loaded length
        import librosa
        original, _ = librosa.load(self.test_wav, sr=16000)
        self.assertLessEqual(len(audio), len(original))


class TestPostprocessText(unittest.TestCase):

    def test_correction_applied(self):
        self.assertEqual(postprocess_text("nino"), "niño")
        self.assertEqual(postprocess_text("rapido"), "rápido")
        self.assertEqual(postprocess_text("senor"), "señor")

    def test_no_correction_needed(self):
        self.assertEqual(postprocess_text("hola mundo"), "hola mundo")

    def test_mixed_text(self):
        result = postprocess_text("el nino corre rapido")
        self.assertEqual(result, "el niño corre rápido")

    def test_empty_string(self):
        self.assertEqual(postprocess_text(""), "")


class TestEvaluate(unittest.TestCase):

    def test_perfect_match(self):
        self.assertAlmostEqual(evaluate("hola mundo", "hola mundo"), 0.0)

    def test_full_mismatch(self):
        score = evaluate("hola mundo", "adios amigo")
        self.assertGreater(score, 0.0)

    def test_partial_match(self):
        score = evaluate("hola mundo", "hola amigo")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


if __name__ == "__main__":
    unittest.main()
