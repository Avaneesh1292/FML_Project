from __future__ import annotations

import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WEBSITE))

from mlp_inference import load_mlp_predictor
from src.feature_extraction import FEATURE_COLUMNS
from src.prediction import decision_scores_for_model, load_predictor, predict_features
from streamlit_app.components import create_decision_score_figure, render_prediction_card
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
        model = bundle["model"]
        selected = pd.DataFrame(
            [[vector[name] for name in bundle["selected_columns"]]],
            columns=bundle["selected_columns"],
        )
        expected_genre = str(model.predict(selected)[0])
        expected_raw = model.decision_function(selected)[0]
        self.assertEqual(result["genre"], expected_genre)
        self.assertEqual(
            list(result["decision_scores"].items()),
            [(str(label), float(score)) for label, score in zip(model.classes_, expected_raw)],
        )
        self.assertEqual(len(result["decision_scores"]), 16)
        self.assertTrue(np.isfinite(list(result["decision_scores"].values())).all())
        self.assertEqual(web_result, result)

    def test_decision_scores_map_class_order_and_preserve_negative_values(self):
        class FakeDecisionModel:
            classes_ = np.array(["Jazz", "Rock", "Folk"])

            def decision_function(self, values):
                self.asserted_shape = values.shape
                return np.array([[-1.25, 0.6, -0.4]])

        model = FakeDecisionModel()
        scores = decision_scores_for_model(model, np.zeros((1, 37)), ["unused"])
        self.assertEqual(model.asserted_shape, (1, 37))
        self.assertEqual(scores, {"Jazz": -1.25, "Rock": 0.6, "Folk": -0.4})

    def test_decision_scores_are_unavailable_without_method_and_reject_bad_dimensions(self):
        class NoDecisionMethod:
            pass

        class MalformedDecisionModel:
            classes_ = np.array(["Jazz", "Rock", "Folk"])

            def decision_function(self, values):
                return np.array([[0.2, 0.8]])

        self.assertIsNone(decision_scores_for_model(NoDecisionMethod(), np.zeros((1, 37)), ["Jazz"]))
        with self.assertRaisesRegex(ValueError, "one score per class"):
            decision_scores_for_model(MalformedDecisionModel(), np.zeros((1, 37)), ["Jazz"])

    def test_decision_score_figure_is_ranked_and_keeps_negative_scores(self):
        figure = create_decision_score_figure(
            {"Jazz": -1.25, "Rock": 0.6, "Folk": -0.4, "Pop": -2.0}
        )
        self.assertEqual(list(figure.data[0].y), ["Rock", "Folk", "Jazz", "Pop"])
        self.assertEqual(list(figure.data[0].x), [0.6, -0.4, -1.25, -2.0])
        self.assertEqual(list(figure.data[0].text), ["0.60", "-0.40", "-1.25", "-2.00"])
        self.assertEqual(figure.data[0].hovertemplate, "%{y}: %{x:.2f}<extra></extra>")
        self.assertEqual(figure.data[0].textfont.size, 12)
        self.assertEqual(figure.data[0].textfont.color, "#F0F0F0")
        self.assertEqual(figure.layout.xaxis.tickformat, ".2f")
        self.assertLess(figure.layout.xaxis.range[0], 0)
        self.assertGreater(figure.layout.xaxis.range[1], 0)

    def test_svm_card_renders_score_label_chart_and_explanation(self):
        scores = {"Rock": 0.6, "Jazz": -1.25, "Folk": -0.4}
        with (
            patch("streamlit_app.components.st.container", return_value=nullcontext()),
            patch("streamlit_app.components.st.caption") as caption,
            patch("streamlit_app.components.st.markdown") as markdown,
            patch("streamlit_app.components.st.plotly_chart") as plotly_chart,
        ):
            render_prediction_card({
                "start": 0.0,
                "end": 30.0,
                "filename": "sample.wav",
                "genre": "Rock",
                "model_name": "RBF-SVM",
                "decision_scores": scores,
            })

        markdown_text = " ".join(str(call.args[0]) for call in markdown.call_args_list)
        caption_text = " ".join(str(call.args[0]) for call in caption.call_args_list)
        self.assertIn("SVM Decision Scores", markdown_text)
        self.assertIn("Decision scores indicate how the SVM separates classes; they are not probabilities or calibrated confidence values.", caption_text)
        self.assertEqual(list(plotly_chart.call_args.args[0].data[0].x), [0.6, -0.4, -1.25])


    def test_app_exposes_selector_and_compact_classification_controls(self):
        app_source = (WEBSITE / "app.py").read_text(encoding="utf-8")
        self.assertIn('options=["RBF-SVM", "PyTorch MLP"]', app_source)
        self.assertIn('"Classify Genre"', app_source)
        self.assertIn('with st.expander("Model details and evaluation"', app_source)
        self.assertIn('key="selected_model"', app_source)
        self.assertIn('"model_name": selected_model', app_source)
        component_source = (WEBSITE / "streamlit_app" / "components.py").read_text(encoding="utf-8")
        self.assertIn('model_name = result.get("model_name", "RBF-SVM")', component_source)
        self.assertIn("r.get('model_name'", component_source)
        self.assertIn("SVM Decision Scores", component_source)
        self.assertIn(
            "Decision scores indicate how the SVM separates classes; they are not probabilities or calibrated confidence values.",
            component_source,
        )
        self.assertIn("MLP softmax outputs · not calibrated confidence", component_source)
        self.assertIn('st.container(key="model-comparison-cards")', component_source)
        style_source = (WEBSITE / "streamlit_app" / "styles.py").read_text(encoding="utf-8")
        self.assertIn(".st-key-model-comparison-cards", style_source)
        self.assertIn(".st-key-prediction-card-svm", style_source)
        self.assertIn(".st-key-prediction-card-mlp", style_source)
        self.assertIn("min-height:31.5rem", style_source)


if __name__ == "__main__":
    unittest.main()
