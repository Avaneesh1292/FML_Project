"""CPU-only loader for the exported Phase 5 MLP inference checkpoint."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class _MLPPredictor:
    def __init__(self, model: Any, scaler: Any, classes: list[str], feature_names: list[str]):
        self.model = model
        self.scaler = scaler
        self.classes_ = np.asarray(classes)
        self.feature_names = list(feature_names)
        self.n_features_in_ = len(feature_names)

    def predict_proba(self, X):
        import torch
        values = self.scaler.transform(X).astype(np.float32, copy=False)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.as_tensor(values, dtype=torch.float32, device="cpu"))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def load_mlp_predictor(path: str | Path) -> dict[str, Any]:
    """Load a tensor-only checkpoint on CPU and validate its feature metadata."""
    try:
        import torch
        from sklearn.preprocessing import StandardScaler
        from src.feature_extraction import FEATURE_COLUMNS
        from src.model_training import TabularMLP
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Unable to load CPU MLP checkpoint '{path}': {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("MLP checkpoint has an unsupported or malformed format")
    selected = payload.get("selected_columns")
    labels = payload.get("labels")
    if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
            or any(name not in FEATURE_COLUMNS for name in selected)):
        raise ValueError("MLP checkpoint selected feature names are invalid for the 74-feature schema")
    if (not isinstance(labels, list) or len(labels) != 16 or len(set(labels)) != len(labels)
            or any(not isinstance(label, str) or not label for label in labels)):
        raise ValueError("MLP checkpoint must contain 16 unique genre labels")
    if payload.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("MLP checkpoint feature schema/order is incompatible")
    architecture = payload.get("architecture", {})
    if int(architecture.get("input_dim", -1)) != len(selected) or int(architecture.get("output_dim", -1)) != len(labels):
        raise ValueError("MLP architecture dimensions do not match saved features and labels")
    scaler_data = payload.get("scaler", {})
    if int(scaler_data.get("n_features_in", -1)) != len(selected):
        raise ValueError("MLP scaler dimension does not match selected features")
    mean = np.asarray(scaler_data.get("mean"), dtype=np.float64)
    scale = np.asarray(scaler_data.get("scale"), dtype=np.float64)
    var = np.asarray(scaler_data.get("var"), dtype=np.float64)
    if any(x.shape != (len(selected),) or not np.isfinite(x).all() for x in (mean, scale, var)) or np.any(scale == 0):
        raise ValueError("MLP scaler statistics are missing or invalid")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.var_ = mean, scale, var
    scaler.n_features_in_ = len(selected)
    scaler.feature_names_in_ = np.asarray(selected, dtype=object)
    scaler.n_samples_seen_ = int(scaler_data["n_samples_seen"])
    model = TabularMLP(len(selected), len(labels), architecture["hidden_layers"],
                       float(architecture["dropout"]), str(architecture["activation"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    predictor = _MLPPredictor(model, scaler, labels, selected)
    bundle = {"schema_version": 1, "feature_columns": list(FEATURE_COLUMNS),
              "selected_columns": selected, "labels": labels,
              "audio_config": payload["audio_config"], "representation": "mutual_information",
              "transformer": None, "model": predictor, "source_sha256": payload["source_sha256"]}
    return bundle
