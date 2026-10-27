from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WEBSITE))

from mlp_inference import load_mlp_predictor
from src.feature_extraction import FEATURE_COLUMNS
from src.prediction import load_predictor, predict_features
from web_utils import predict_feature_vector


class TestWebsiteDualModel(unittest.TestCase):
    def test_cpu_mlp_export_schema_order_and_probabilities(self):
        bundle = load_mlp_predictor(WEBSITE / "models" / "phase5_mutual_information_mlp_cpu.pt")
        self.assertEqual(bundle["feature_columns"], FEATURE_COLUMNS)
        self.assertEqual(len(bundle["feature_columns"]), 74)
        self.assertEqual(len(bundle["selected_columns"]), 37)
        self.assertEqual(bundle["model"].feature_names, bundle["selected_columns"])
        self.assertFalse(bundle["model"].model.training)
        row = pd.DataFrame(np.zeros((1, 37), dtype=np.float32), columns=bundle["selected_columns"])
        probs = bundle["model"].predict_proba(row)
        self.assertEqual(probs.shape, (1, 16))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue(np.isclose(probs.sum(), 1.0, atol=1e-6))
        self.assertIn(bundle["model"].predict(row)[0], bundle["labels"])


    def test_phase5b_svm_predictor_still_uses_saved_bundle(self):
        bundle = load_predictor(ROOT / "models" / "phase5b_svm" / "phase5b_svm__mutual_information.joblib")
        mlp = load_mlp_predictor(WEBSITE / "models" / "phase5_mutual_information_mlp_cpu.pt")
        self.assertEqual(bundle["selected_columns"], mlp["selected_columns"])
        vector = {name: 0.0 for name in FEATURE_COLUMNS}
        result = predict_features(vector, bundle)
        web_result = predict_feature_vector(vector, bundle)
        self.assertIn(result["genre"], bundle["labels"])
        self.assertIsNone(result["probabilities"])
        self.assertEqual(web_result, result)


    def test_app_exposes_selector_and_compact_classification_controls(self):
        app_source = (WEBSITE / "app.py").read_text(encoding="utf-8")
        self.assertIn('options=["RBF-SVM", "PyTorch MLP"]', app_source)
        self.assertIn('"Classify Genre"', app_source)
        self.assertIn('with st.expander("Model details and evaluation"', app_source)
        self.assertIn('key="selected_model"', app_source)
        self.assertIn('"model_name": selected_model', app_source)
        component_source = (WEBSITE / "streamlit_app" / "components.py").read_text(encoding="utf-8")
        self.assertIn("result.get('model_name'", component_source)
        self.assertIn("r.get('model_name'", component_source)


if __name__ == "__main__":
    unittest.main()
