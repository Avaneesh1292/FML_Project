"""Small safety tests for the isolated Phase 5B SVM follow-up."""
from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from src.dimensionality_reduction import PCATransformer
from src.svm_followup import (
    assert_fresh_report_path,
    atomic_json,
    build_conventional_svc,
    candidate_grid,
    fit_candidate,
    load_or_create_checkpoint,
    save_new_bundle,
    select_on_validation,
    validate_representation_cache,
)


class SvmFollowupTests(unittest.TestCase):
    def test_nine_candidate_grid(self):
        grid = candidate_grid()
        self.assertEqual(len(grid), 9)
        self.assertEqual(len({(x["C"], x["gamma"]) for x in grid}), 9)
        self.assertEqual([x["C"] for x in grid[:3]], [0.1, 0.1, 0.1])
        self.assertEqual([x["gamma"] for x in grid[:3]], ["scale", 0.001, 0.1])

    def test_conventional_svc_fits_full_passed_training_partition(self):
        rng = np.random.default_rng(21)
        X = pd.DataFrame(rng.normal(size=(24, 4)), columns=list("abcd"))
        y = np.array(["a"] * 12 + ["b"] * 12)
        pipeline = build_conventional_svc({"C": 1.0, "gamma": "scale"})
        pipeline.fit(X, y)
        self.assertEqual(int(pipeline.named_steps["standardscaler"].n_samples_seen_), 24)
        self.assertIsInstance(pipeline.steps[-1][1], SVC)
        self.assertEqual(pipeline.steps[-1][1].kernel, "rbf")
        self.assertFalse(pipeline.steps[-1][1].probability)
        self.assertIsNone(pipeline.steps[-1][1].class_weight)

    def test_validation_selection_accepts_only_train_and_validation_inputs(self):
        class FakeEstimator:
            fitted_ids = []

            def __init__(self, params):
                self.params = params

            def fit(self, X, y):
                self.fitted_ids.append(tuple(X[:, 0].tolist()))
                self.n_fit_ = len(y)
                return self

            def predict(self, X):
                if self.params["C"] == 1:
                    return np.asarray(["a", "b", "a", "b"])
                return np.asarray(["a", "a", "a", "a"])

        Xtr = np.arange(12, dtype=float).reshape(6, 2)
        ytr = np.array(["a", "b", "a", "b", "a", "b"])
        Xva = np.arange(8, dtype=float).reshape(4, 2)
        yva = np.array(["a", "b", "a", "b"])
        before = list(FakeEstimator.fitted_ids)
        winner, rows = select_on_validation(
            Xtr, ytr, Xva, yva, ["a", "b"],
            candidates=[{"C": 0.1, "gamma": "scale"}, {"C": 1, "gamma": "scale"}],
            pipeline_factory=FakeEstimator,
        )
        self.assertEqual(winner["C"], 1)
        self.assertEqual([r["training_rows"] for r in rows], [6, 6])
        self.assertEqual([r["validation_rows"] for r in rows], [4, 4])
        self.assertEqual(FakeEstimator.fitted_ids[len(before):], [tuple(Xtr[:, 0]), tuple(Xtr[:, 0])])
        self.assertNotIn("test", inspect.signature(select_on_validation).parameters)
        self.assertTrue(all("validation_metrics" in row for row in rows))

    def _cache_fixture(self):
        rng = np.random.default_rng(12)
        cols = list("abcd")
        full_train = pd.DataFrame(rng.normal(size=(12, 4)), columns=cols)
        full_validation = pd.DataFrame(rng.normal(size=(6, 4)), columns=cols)
        mi_cols, rf_cols = cols[:2], cols[1:]
        pca = PCATransformer(0.95).fit(full_train)
        reps = {
            "full": (full_train, full_validation, cols, None),
            "mutual_information": (full_train[mi_cols], full_validation[mi_cols], mi_cols, None),
            "random_forest": (full_train[rf_cols], full_validation[rf_cols], rf_cols, None),
            "pca_95": (pca.transform(full_train), pca.transform(full_validation), cols, pca),
        }
        return {
            "run_signature": "fixture", "training_ids": list(range(12)),
            "validation_ids": list(range(100, 106)), "feature_columns": cols,
            "representations": reps,
            "representation_info": {
                "full": {"selected_features": cols},
                "mutual_information": {"selected_features": mi_cols},
                "random_forest": {"selected_features": rf_cols},
                "pca_95": {"retained_components": pca.pca.n_components_},
            },
        }

    def test_representation_dimensions_and_saved_pca_are_compatible(self):
        cache = self._cache_fixture()
        validate_representation_cache(cache, run_signature="fixture",
                                      training_ids=list(range(12)),
                                      validation_ids=list(range(100, 106)),
                                      feature_columns=list("abcd"),
                                      expected_train_rows=12, expected_validation_rows=6)
        self.assertEqual(cache["representations"]["full"][0].shape[1], 4)
        self.assertEqual(cache["representations"]["mutual_information"][0].shape[1], 2)
        self.assertEqual(cache["representations"]["random_forest"][0].shape[1], 3)
        self.assertEqual(cache["representations"]["pca_95"][0].shape[1],
                         cache["representations"]["pca_95"][3].pca.n_components_)
        bad = self._cache_fixture()
        bad["representations"]["mutual_information"] = (
            bad["representations"]["mutual_information"][0].iloc[:, :1],
            bad["representations"]["mutual_information"][1], ["a", "b"], None)
        with self.assertRaises(ValueError):
            validate_representation_cache(bad, run_signature="fixture",
                                          training_ids=list(range(12)),
                                          validation_ids=list(range(100, 106)),
                                          feature_columns=list("abcd"),
                                          expected_train_rows=12, expected_validation_rows=6)

    def test_checkpoint_recovery_signature_and_uncheckpointed_output_protection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp); results = root / "results"; models = root / "models"
            cp = results / "phase5b_svm_checkpoint.json"
            state = load_or_create_checkpoint(cp, run_signature="abc",
                                              results_dir=results, models_dir=models)
            state["candidate_results"].append({"representation": "full", "candidate_index": 0})
            atomic_json(cp, state)
            resumed = load_or_create_checkpoint(cp, run_signature="abc",
                                                results_dir=results, models_dir=models)
            self.assertEqual(resumed["candidate_results"], state["candidate_results"])
            with self.assertRaises(ValueError):
                load_or_create_checkpoint(cp, run_signature="different",
                                          results_dir=results, models_dir=models)
            stray = root / "stray_results"; stray.mkdir(); (stray / "old.csv").write_text("x")
            with self.assertRaises(FileExistsError):
                load_or_create_checkpoint(root / "other.json", run_signature="x",
                                          results_dir=stray, models_dir=root / "new_models")

    def test_bundle_and_report_overwrite_protection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / "model.joblib"
            save_new_bundle(path, {"x": 1})
            with self.assertRaises(FileExistsError):
                save_new_bundle(path, {"x": 2})
            with self.assertRaises(FileExistsError):
                assert_fresh_report_path(path)


if __name__ == "__main__":
    unittest.main()
