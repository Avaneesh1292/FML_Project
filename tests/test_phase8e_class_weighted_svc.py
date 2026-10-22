"""Small safeguards for the isolated Phase 8E validation experiment."""
import unittest

import numpy as np

from scripts.run_phase8e_class_weighted_svc import build_balanced_svc
from src.svm_followup import build_conventional_svc, fit_candidate


class Phase8EClassWeightedSVCTests(unittest.TestCase):
    def test_only_class_weight_differs(self):
        params = {"C": 10.0, "gamma": "scale"}
        baseline = build_conventional_svc(params)
        balanced = build_balanced_svc(params)
        self.assertEqual(list(baseline.named_steps), list(balanced.named_steps))
        self.assertEqual(baseline.named_steps["svc"].get_params(), {
            **balanced.named_steps["svc"].get_params(), "class_weight": None
        })
        self.assertIsNone(baseline.named_steps["svc"].class_weight)
        self.assertEqual(balanced.named_steps["svc"].class_weight, "balanced")

    def test_validation_metrics_use_configured_all_class_order(self):
        classes = [f"g{i}" for i in range(16)]
        rng = np.random.default_rng(7)
        x_train = rng.normal(size=(32, 4))
        y_train = np.asarray([classes[i] for i in range(16)] * 2, dtype=object)
        x_validation = rng.normal(size=(16, 4))
        y_validation = np.asarray(classes, dtype=object)
        _, record = fit_candidate(x_train, y_train, x_validation, y_validation,
                                  classes, {"C": 10.0, "gamma": "scale"},
                                  pipeline_factory=build_balanced_svc)
        metrics = record["validation_metrics"]
        self.assertEqual(record["training_rows"], 32)
        self.assertEqual(record["validation_rows"], 16)
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (16, 16))
        self.assertEqual(sum(row["support"] for row in metrics["per_class"].values()), 16)


if __name__ == "__main__":
    unittest.main()
