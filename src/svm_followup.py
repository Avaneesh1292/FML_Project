"""Isolated Phase 5B conventional RBF SVC helpers."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import os
import tempfile
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEFAULT_C = (0.1, 1.0, 10.0)
DEFAULT_GAMMA = ("scale", 0.001, 0.1)


def candidate_grid(c_values=DEFAULT_C, gamma_values=DEFAULT_GAMMA) -> list[dict[str, Any]]:
    """Return a deterministic C-major grid (nine candidates by default)."""
    if not c_values or not gamma_values:
        raise ValueError("SVM C and gamma search values must both be non-empty")
    if any(float(c) <= 0 for c in c_values):
        raise ValueError("Every SVM C value must be positive")
    if any(not (isinstance(g, str) and g == "scale") and float(g) <= 0
           for g in gamma_values):
        raise ValueError("Numeric SVM gamma values must be positive; string gamma must be 'scale'")
    return [{"C": float(c), "gamma": g} for c, g in product(c_values, gamma_values)]


def build_conventional_svc(params: dict[str, Any], *, cache_size: float = 200.0):
    """One scaled RBF SVC, with probability calibration and class weighting off."""
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=float(params["C"]), gamma=params["gamma"],
            class_weight=None, probability=False, decision_function_shape="ovr",
            cache_size=float(cache_size)),
    )


def fit_candidate(X_train, y_train, X_validation, y_validation,
                  class_order: list[str], params: dict[str, Any],
                  pipeline_factory: Callable[[dict[str, Any]], Any] = build_conventional_svc
                  ) -> tuple[Any, dict[str, Any]]:
    """Fit one candidate on train rows and record validation-only metrics."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    estimator = pipeline_factory(dict(params))
    start = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predictions = estimator.predict(X_validation)
    inference_seconds = time.perf_counter() - start
    f = f1_score(y_validation, predictions, labels=class_order,
                 average="macro", zero_division=0)
    # Use sklearn's classification report to persist all class metrics and the
    # explicit class ordering; aggregate metrics are recomputed from class labels.
    report = classification_report(y_validation, predictions, labels=class_order,
                                   output_dict=True, zero_division=0)
    class_p = np.asarray([report[label]["precision"] for label in class_order])
    class_r = np.asarray([report[label]["recall"] for label in class_order])
    class_f = np.asarray([report[label]["f1-score"] for label in class_order])
    metrics = {
        "accuracy": float(accuracy_score(y_validation, predictions)),
        "macro_precision": float(class_p.mean()),
        "macro_recall": float(class_r.mean()),
        "macro_f1": float(class_f.mean()),
        "inference_seconds": float(inference_seconds),
        "inference_ms_per_track": float(inference_seconds * 1000 / max(1, len(predictions))),
        "confusion_matrix": confusion_matrix(y_validation, predictions,
                                              labels=class_order).tolist(),
        "per_class": {label: report[label] for label in class_order},
    }
    if not np.isclose(metrics["macro_f1"], float(f), rtol=1e-10, atol=1e-12):
        raise RuntimeError("Validation macro-F1 calculation is inconsistent")
    return estimator, {**params, "validation_metrics": metrics,
                       "training_seconds": float(fit_seconds),
                       "training_rows": int(len(X_train)),
                       "validation_rows": int(len(X_validation))}


def select_on_validation(X_train, y_train, X_validation, y_validation,
                         class_order: list[str], candidates=None,
                         pipeline_factory: Callable[[dict[str, Any]], Any] = build_conventional_svc
                         ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit all candidates on train rows and select strictly by validation macro-F1."""
    candidates = list(candidates if candidates is not None else candidate_grid())
    if not candidates:
        raise ValueError("At least one SVM candidate is required")
    if len(X_train) != len(y_train) or len(X_validation) != len(y_validation):
        raise ValueError("Feature and label row counts differ")
    if set(pd.unique(y_train)) != set(class_order):
        raise ValueError("Training labels do not cover the configured class order")
    if set(pd.unique(y_validation)) != set(class_order):
        raise ValueError("Validation labels do not cover the configured class order")
    rows: list[dict[str, Any]] = []
    for index, params in enumerate(candidates):
        _, record = fit_candidate(X_train, y_train, X_validation, y_validation,
                                  class_order, params, pipeline_factory)
        rows.append({"candidate_index": index, **record,
                     "validation_macro_f1": record["validation_metrics"]["macro_f1"]})
    # max preserves the first candidate on ties given the saved candidate order.
    best = max(rows, key=lambda row: row["validation_macro_f1"])
    return dict(best), rows


def validate_representation_cache(cache: dict, *, run_signature: str,
                                  training_ids: list[int], validation_ids: list[int],
                                  feature_columns: list[str], expected_train_rows: int,
                                  expected_validation_rows: int) -> None:
    """Refuse stale/misordered cache or PCA state before candidate fitting."""
    if cache.get("run_signature") != run_signature:
        raise ValueError("Phase 5 representation cache signature is stale")
    if cache.get("training_ids") != training_ids:
        raise ValueError("Cached training IDs/order differ from the official training partition")
    if cache.get("validation_ids") != validation_ids:
        raise ValueError("Cached validation IDs/order differ from the official validation partition")
    if cache.get("feature_columns") != feature_columns:
        raise ValueError("Cached feature schema/order differs from the configured schema")
    reps = cache.get("representations", {})
    info = cache.get("representation_info", {})
    expected = {"full", "mutual_information", "random_forest", "pca_95"}
    if set(reps) != expected or set(info) != expected:
        raise ValueError("Representation cache does not contain exactly the four expected variants")
    for name, (X_train, X_validation, columns, transformer) in reps.items():
        if X_train.shape[0] != expected_train_rows or X_validation.shape[0] != expected_validation_rows:
            raise ValueError(f"Cached {name} row counts do not match official train/validation sizes")
        if name == "full":
            if list(columns) != feature_columns or X_train.shape[1] != len(feature_columns) or transformer is not None:
                raise ValueError("Full feature cache is incompatible with the configured schema")
        elif name in ("mutual_information", "random_forest"):
            if list(columns) != info[name].get("selected_features"):
                raise ValueError(f"Cached {name} selected feature order differs from its metadata")
            if X_train.shape[1] != len(columns) or X_validation.shape[1] != len(columns):
                raise ValueError(f"Cached {name} dimensions disagree with selected columns")
            if not set(columns).issubset(feature_columns):
                raise ValueError(f"Cached {name} contains columns outside the original schema")
        else:
            if transformer is None or list(transformer.feature_names_in_ or []) != feature_columns:
                raise ValueError("Cached PCA transformer has an incompatible original feature schema")
            if int(transformer.pca.n_samples_) != expected_train_rows:
                raise ValueError("Cached PCA was not fitted on exactly the training partition")
            if int(transformer.scaler.n_samples_seen_) != expected_train_rows:
                raise ValueError("Cached PCA scaler was not fitted on exactly the training partition")
            if X_train.shape[1] != int(transformer.pca.n_components_):
                raise ValueError("Cached PCA component dimension does not match fitted PCA")
            full_train, full_validation = reps["full"][0], reps["full"][1]
            if not np.allclose(transformer.transform(full_train), X_train, rtol=1e-6, atol=1e-6):
                raise ValueError("Cached PCA training values differ from saved fitted transform")
            if not np.allclose(transformer.transform(full_validation), X_validation,
                               rtol=1e-6, atol=1e-6):
                raise ValueError("Cached PCA validation values differ from saved fitted transform")


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try: os.unlink(temp_name)
        except OSError: pass
        raise


def load_or_create_checkpoint(path: str | Path, *, run_signature: str,
                              results_dir: str | Path, models_dir: str | Path) -> dict:
    """Resume only matching runs; refuse uncheckpointed output that could be overwritten."""
    path = Path(path); results_dir = Path(results_dir); models_dir = Path(models_dir)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("run_signature") != run_signature:
            raise ValueError("Existing Phase 5B checkpoint belongs to different inputs/configuration")
        if state.get("test_in_progress"):
            raise RuntimeError("Exploratory test inference was interrupted; refusing to repeat it")
        return state
    occupied = [d for d in (results_dir, models_dir)
                if d.exists() and any(d.rglob("*"))]
    if occupied:
        raise FileExistsError(f"Refusing to overwrite uncheckpointed Phase 5B output: {occupied}")
    results_dir.mkdir(parents=True, exist_ok=True); models_dir.mkdir(parents=True, exist_ok=True)
    state = {"version": 1, "run_signature": run_signature, "stage": "validation_search",
             "candidate_results": [], "selected": {},
             "exploratory_test_results": {}, "test_in_progress": None}
    atomic_json(path, state)
    return state


def save_new_bundle(path: str | Path, bundle: dict) -> None:
    """Atomically save a bundle without replacing any pre-existing file."""
    import joblib
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing model bundle: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".joblib.tmp", dir=path.parent)
    os.close(fd)
    try:
        joblib.dump(bundle, temp_name)
        if path.exists():
            raise FileExistsError(f"Bundle appeared while saving; refusing overwrite: {path}")
        os.replace(temp_name, path)
    except Exception:
        try: os.unlink(temp_name)
        except OSError: pass
        raise


def assert_fresh_report_path(path: str | Path) -> None:
    if Path(path).exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 5B report: {path}")
