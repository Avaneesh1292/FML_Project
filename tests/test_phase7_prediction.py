"""Focused safety checks for the saved Phase 5B prediction bundle."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import soundfile as sf

from src.audio_preprocessing import AudioProcessingError
from src.feature_extraction import FEATURE_COLUMNS
from src.prediction import load_predictor, predict_audio

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "models/phase5b_svm/phase5b_svm__mutual_information.joblib"


class Phase7PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_predictor(BUNDLE_PATH)

    def test_phase5b_mi_bundle_schema_order_and_dimensions(self):
        cache = joblib.load(ROOT / "results/metrics/representation_fit_checkpoint.joblib")
        expected = cache["representation_info"]["mutual_information"]["selected_features"]
        self.assertEqual(len(FEATURE_COLUMNS), 74)
        self.assertEqual(self.bundle["representation"], "mutual_information")
        self.assertEqual(len(self.bundle["labels"]), 16)
        self.assertEqual(len(self.bundle["selected_columns"]), 37)
        self.assertEqual(self.bundle["selected_columns"], expected)
        self.assertEqual(
            list(self.bundle["model"].named_steps["standardscaler"].feature_names_in_), expected
        )
        self.assertEqual(self.bundle["model"].named_steps["svc"].n_features_in_, 37)

    def test_rejects_duplicate_or_unknown_selected_columns(self):
        for selected in (
            self.bundle["selected_columns"][:-1] + [self.bundle["selected_columns"][0]],
            self.bundle["selected_columns"][:-1] + ["not_a_feature"],
        ):
            malformed = dict(self.bundle, selected_columns=selected)
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = Path(directory) / "malformed.joblib"
                joblib.dump(malformed, path)
                with self.assertRaises(ValueError):
                    load_predictor(path)

    def test_rejects_incompatible_labels_and_input_dimension(self):
        malformed_labels = dict(self.bundle, labels=list(reversed(self.bundle["labels"])))
        malformed_dimension = dict(
            self.bundle, selected_columns=self.bundle["selected_columns"][:-1]
        )
        for malformed in (malformed_labels, malformed_dimension):
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = Path(directory) / "malformed.joblib"
                joblib.dump(malformed, path)
                with self.assertRaises(ValueError):
                    load_predictor(path)

    def test_missing_audio_returns_clear_error(self):
        with self.assertRaisesRegex(AudioProcessingError, "Audio file not found"):
            predict_audio(ROOT / "missing_phase7_audio.wav", self.bundle)

    def test_corrupt_audio_returns_clear_error(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "corrupt.wav"
            path.write_bytes(b"not an audio stream")
            fake_librosa = SimpleNamespace(
                load=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("decode failed"))
            )
            with patch.dict("sys.modules", {"librosa": fake_librosa}):
                with self.assertRaisesRegex(AudioProcessingError, "Unable to decode"):
                    predict_audio(path, self.bundle)

    def test_too_short_audio_returns_clear_error(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "short.wav"
            sf.write(path, np.zeros(1102, dtype=np.float32), 22050)  # about 0.05 seconds
            with self.assertRaisesRegex(AudioProcessingError, "Track too short"):
                predict_audio(path, self.bundle)


if __name__ == "__main__":
    unittest.main()
