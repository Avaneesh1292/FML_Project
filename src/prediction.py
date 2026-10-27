"""Save and load inference bundles without refitting audio transforms."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .feature_extraction import FEATURE_COLUMNS, extract_feature_vector


def save_predictor(path: str | Path, model: Any, feature_names: list[str], labels: list[str],
                   audio_config: dict[str, Any], representation: str = "full",
                   transformer: Any = None) -> None:
    bundle = {"schema_version": 1, "feature_columns": FEATURE_COLUMNS,
              "selected_columns": feature_names, "labels": labels,
              "audio_config": audio_config, "representation": representation,
              "transformer": transformer, "model": model}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_predictor(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ValueError(f"Unable to load predictor bundle '{path}': {type(exc).__name__}: {exc}") from exc
    if not isinstance(bundle, dict):
        raise ValueError("Saved predictor bundle must be a dictionary")
    if bundle.get("schema_version") != 1:
        raise ValueError("Saved predictor bundle has an unsupported schema version")
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Saved predictor feature schema/order is incompatible with this code")

    selected = bundle.get("selected_columns")
    if (not isinstance(selected, (list, tuple)) or not selected
            or any(not isinstance(name, str) for name in selected)):
        raise ValueError("Saved predictor selected_columns must be a non-empty list of feature names")
    if len(set(selected)) != len(selected):
        raise ValueError("Saved predictor selected_columns contains duplicate feature names")
    unknown = [name for name in selected if name not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"Saved predictor contains features outside the 74-feature schema: {unknown}")

    labels = bundle.get("labels")
    if (not isinstance(labels, (list, tuple)) or len(labels) < 2
            or any(not isinstance(label, str) or not label for label in labels)):
        raise ValueError("Saved predictor labels must contain at least two non-empty genre names")
    if len(set(labels)) != len(labels):
        raise ValueError("Saved predictor labels contain duplicates")

    model = bundle.get("model")
    if model is None or not callable(getattr(model, "predict", None)):
        raise ValueError("Saved predictor bundle does not contain a fitted model with predict()")
    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        raise ValueError("Saved predictor model does not expose fitted classes_")
    model_labels = [str(label) for label in np.asarray(model_classes).tolist()]
    if model_labels != list(labels):
        raise ValueError("Saved predictor genre labels/order do not match the fitted model classes")

    transformer = bundle.get("transformer")
    expected_width = len(selected)
    if transformer is not None:
        transformer_columns = getattr(transformer, "feature_names_in_", None)
        if transformer_columns is not None and list(transformer_columns) != list(selected):
            raise ValueError("Saved transformer input feature order does not match selected_columns")
        pca = getattr(transformer, "pca", None)
        component_count = getattr(pca, "n_components_", None)
        if component_count is not None:
            expected_width = int(component_count)
        else:
            # Infer the output width by transforming one neutral row. This
            # validates the fitted object without fitting or changing it.
            sample = pd.DataFrame(np.zeros((1, len(selected))), columns=list(selected))
            try:
                transformed = transformer.transform(sample)
            except Exception as exc:
                raise ValueError(f"Saved transformer cannot transform its declared input schema: {exc}") from exc
            if getattr(transformed, "ndim", 0) != 2 or transformed.shape[0] != 1:
                raise ValueError("Saved transformer must return a two-dimensional single-row result")
            expected_width = int(transformed.shape[1])

    # sklearn estimators/pipelines expose n_features_in_; the project MLP
    # wrapper exposes the same fitted width through its training scaler.
    dimensions: list[int] = []
    direct_width = getattr(model, "n_features_in_", None)
    if direct_width is not None:
        dimensions.append(int(direct_width))
    fitted_scaler = getattr(model, "scaler", None)
    scaler_width = getattr(fitted_scaler, "n_features_in_", None)
    if scaler_width is not None:
        dimensions.append(int(scaler_width))
    fitted_input_names: list[list[str]] = []
    scaler_names = getattr(fitted_scaler, "feature_names_in_", None)
    if scaler_names is not None:
        fitted_input_names.append([str(name) for name in scaler_names])
    named_steps = getattr(model, "named_steps", {})
    if named_steps:
        for step in named_steps.values():
            step_width = getattr(step, "n_features_in_", None)
            if step_width is not None:
                dimensions.append(int(step_width))
            step_names = getattr(step, "feature_names_in_", None)
            if step_names is not None:
                fitted_input_names.append([str(name) for name in step_names])
    if transformer is None:
        model_names = getattr(model, "feature_names", None)
        if model_names is None:
            model_names = getattr(model, "feature_names_in_", None)
        if model_names is not None:
            fitted_input_names.append([str(name) for name in model_names])
        if any(names != list(selected) for names in fitted_input_names):
            raise ValueError("Saved model fitted feature order does not match selected_columns")
    if not dimensions:
        raise ValueError("Saved predictor model does not expose its fitted input dimension")
    if any(width != expected_width for width in dimensions):
        raise ValueError(
            f"Saved predictor input dimension mismatch: expected {expected_width} transformed features, "
            f"model reports {sorted(set(dimensions))}"
        )

    audio_config = bundle.get("audio_config")
    required_audio = {"sample_rate", "mono", "n_fft", "hop_length", "n_mfcc", "n_chroma", "roll_percent"}
    if not isinstance(audio_config, dict) or not required_audio.issubset(audio_config):
        raise ValueError("Saved predictor audio_config is missing required feature-extraction settings")
    return bundle


def predict_audio(path: str | Path, bundle: dict[str, Any]) -> dict[str, Any]:
    features = extract_feature_vector(path, bundle["audio_config"])
    return predict_features(features, bundle)


def predict_features(features: dict[str, float], bundle: dict[str, Any]) -> dict[str, Any]:
    """Run a validated predictor on one feature vector using its saved order."""
    if list(features) != FEATURE_COLUMNS:
        missing = [name for name in FEATURE_COLUMNS if name not in features]
        extra = [name for name in features if name not in FEATURE_COLUMNS]
        raise ValueError(f"Feature vector does not match the 74-feature schema (missing={missing}, extra={extra})")
    frame = pd.DataFrame([[features[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    frame = frame.loc[:, bundle["selected_columns"]]
    transformer = bundle.get("transformer")
    values = transformer.transform(frame) if transformer is not None else frame
    model = bundle["model"]
    predicted = str(model.predict(values)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(values)[0]
        classes = getattr(model, "classes_", bundle["labels"])
        probabilities = {str(label): float(probability) for label, probability in zip(classes, raw)}
    return {"genre": predicted, "probabilities": probabilities}

