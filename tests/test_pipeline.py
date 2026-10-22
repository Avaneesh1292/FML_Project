"""Lightweight tests; intentionally no full audio extraction or training."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import soundfile as sf
from src.audio_preprocessing import AudioProcessingError, load_audio
from src.data_loading import load_track_metadata, official_partitions, track_audio_path
from src.dimensionality_reduction import PCATransformer
from src.evaluation import evaluate_predictions
from src.feature_extraction import FEATURE_COLUMNS, extract_feature_vector
from src.feature_selection import mutual_information_ranking, random_forest_ranking, select_top_features
from src.model_training import train_mlp, train_sklearn
from src.prediction import load_predictor, predict_audio, save_predictor
from src.utils import load_config

ROOT = Path(__file__).resolve().parents[1]

class MetadataTests(unittest.TestCase):
    def test_track_id_path_mapping(self):
        self.assertEqual(track_audio_path("audio", 2), Path("audio/000/000002.mp3"))
        with self.assertRaises(ValueError):
            track_audio_path("audio", -1)

    def test_medium_metadata_official_splits(self):
        table = load_track_metadata(ROOT / "fma_metadata/tracks.csv", ROOT / "fma_medium", subset="medium")
        self.assertEqual(len(table), 17_000)
        self.assertEqual(table["genre"].isna().sum(), 0)
        self.assertIsInstance(table.iloc[0]["audio_path"], str)
        self.assertIsInstance(table.iloc[0]["genre"], str)
        parts = official_partitions(table)
        self.assertEqual({k: len(v) for k, v in parts.items()},
                         {"training": 13_522, "validation": 1_705, "test": 1_773})

class AudioFeatureTests(unittest.TestCase):
    def test_missing_and_corrupt_audio(self):
        with self.assertRaises(AudioProcessingError):
            load_audio(ROOT / "does-not-exist.mp3")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            bad = Path(directory) / "broken.mp3"
            bad.write_bytes(b"not an audio stream")
            with patch.dict("sys.modules", {"librosa": SimpleNamespace(load=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("decode error")))}):
                with self.assertRaises(AudioProcessingError):
                    load_audio(bad)

    def test_feature_schema_and_numeric_values(self):
        fake = SimpleNamespace(feature=SimpleNamespace(
            mfcc=lambda **k: np.ones((20, 5)), chroma_stft=lambda **k: np.ones((12, 5)),
            zero_crossing_rate=lambda *a, **k: np.ones((1, 5)),
            spectral_centroid=lambda **k: np.ones((1, 5)), spectral_bandwidth=lambda **k: np.ones((1, 5)),
            spectral_rolloff=lambda **k: np.ones((1, 5)), rms=lambda **k: np.ones((1, 5))))
        with patch("src.feature_extraction.load_audio", return_value=(np.ones(4096, dtype=np.float32), 22050)), \
             patch.dict("sys.modules", {"librosa": fake}):
            cfg = load_config(ROOT / "config/config.yaml")
            vector = extract_feature_vector(ROOT / "unused.wav", cfg["audio"])
        self.assertEqual(list(vector), FEATURE_COLUMNS)
        self.assertEqual(len(vector), 74)
        self.assertTrue(np.isfinite(list(vector.values())).all())
        broken = SimpleNamespace(feature=SimpleNamespace(**{**fake.feature.__dict__,
            "mfcc": lambda **k: np.full((20, 5), np.nan)}))
        with patch("src.feature_extraction.load_audio", return_value=(np.ones(4096, dtype=np.float32), 22050)), \
             patch.dict("sys.modules", {"librosa": broken}):
            with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                extract_feature_vector(ROOT / "unused.wav", cfg["audio"])

class TransformAndModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        cls.X = pd.DataFrame(rng.normal(size=(36, 8)), columns=[f"f{i}" for i in range(8)])
        cls.y = pd.Series(["a", "b", "c"] * 12)

    def test_feature_ranking(self):
        ranking = mutual_information_ranking(self.X.iloc[:27], self.y.iloc[:27])
        names = select_top_features(ranking, 3)
        self.assertEqual(len(names), 3)
        self.assertTrue(set(names).issubset(self.X.columns))
        forest_ranking = random_forest_ranking(self.X.iloc[:27], self.y.iloc[:27], n_estimators=10)
        self.assertEqual(len(select_top_features(forest_ranking, 2)), 2)

    def test_pca_train_fit_and_schema_guard(self):
        pca = PCATransformer(.90).fit(self.X.iloc[:27])
        self.assertEqual(pca.transform(self.X.iloc[27:]).shape[0], 9)
        np.testing.assert_allclose(pca.scaler.mean_, self.X.iloc[:27].mean().to_numpy())
        self.assertGreaterEqual(pca.explained_variance_ratio_.sum(), .90)
        with self.assertRaises(ValueError):
            pca.transform(self.X.iloc[27:][list(reversed(self.X.columns))])

    def test_sklearn_model_and_metrics(self):
        cfg = load_config(ROOT / "config/config.yaml")
        cfg["models"]["random_forest"]["n_estimators"] = 10
        for name in ("logistic_regression", "svm", "random_forest"):
            model, elapsed = train_sklearn(name, self.X.iloc[:27], self.y.iloc[:27], cfg, 3)
            pred = model.predict(self.X.iloc[27:])
            if name == "svm":
                self.assertEqual(model.predict_proba(self.X.iloc[27:]).shape, (9, 3))
            metrics = evaluate_predictions(self.y.iloc[27:], pred, ["a", "b", "c"], elapsed, len(pred))
            self.assertEqual(len(pred), 9)
            self.assertIn("macro_f1", metrics)
            self.assertEqual(len(metrics["confusion_matrix"]), 3)

    def test_mlp_cpu_and_saved_prediction(self):
        cfg = load_config(ROOT / "config/config.yaml")
        cfg["models"]["mlp"].update({"hidden_layers": [12], "dropout": 0., "max_epochs": 3,
                                      "patience": 2, "batch_size": 9})
        rng = np.random.default_rng(12)
        X = pd.DataFrame(rng.normal(size=(36, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        fitted = train_mlp(X.iloc[:27], self.y.iloc[:27], X.iloc[27:], self.y.iloc[27:],
                           cfg, seed=5, device="cpu")
        self.assertGreaterEqual(fitted.best_epoch, 1)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            audio = Path(directory) / "sample.wav"
            audio.write_bytes(b"inference extractor is mocked")
            artifact = Path(directory) / "mlp.joblib"
            save_predictor(artifact, fitted, FEATURE_COLUMNS, ["a", "b", "c"], cfg["audio"])
            fake_vector = {name: float(i) for i, name in enumerate(FEATURE_COLUMNS)}
            with patch("src.prediction.extract_feature_vector", return_value=fake_vector):
                result = predict_audio(audio, load_predictor(artifact))
            self.assertIn(result["genre"], {"a", "b", "c"})

if __name__ == "__main__":
    unittest.main()

