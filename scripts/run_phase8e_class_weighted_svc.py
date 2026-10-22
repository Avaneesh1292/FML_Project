"""Run the isolated Phase 8E class-weighted MI SVC validation comparison."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_svm_followup import _labels_for_splits  # train/validation filter only
from src.feature_extraction import FEATURE_COLUMNS
from src.svm_followup import build_conventional_svc, fit_candidate, save_new_bundle

REPRESENTATION = "mutual_information"
PARAMS = {"C": 10.0, "gamma": "scale"}
OUTPUT = ROOT / "results/phase8e_class_weighted_svc"
MODEL_OUTPUT = ROOT / "models/phase8e_class_weighted_svc"


def build_balanced_svc(params: dict):
    """Mirror the Phase 5B pipeline, changing class_weight only."""
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=float(params["C"]), gamma=params["gamma"],
            class_weight="balanced", probability=False,
            decision_function_shape="ovr", cache_size=200.0),
    )


def protected_hashes() -> dict[str, str]:
    """Hash existing Phase 5/5B outputs, without parsing their test metrics."""
    targets = [ROOT / "results/metrics", ROOT / "results/figures",
               ROOT / "results/phase5b_svm", ROOT / "models/phase5b_svm",
               ROOT / "results/reports/phase5b_svm_followup.md"]
    paths: set[Path] = set()
    for target in targets:
        if target.is_file():
            paths.add(target)
        elif target.is_dir():
            paths.update(path for path in target.rglob("*") if path.is_file())
    result = {}
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[path.relative_to(ROOT).as_posix()] = digest
    return result


def load_train_validation():
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    feature_path = ROOT / config["paths"]["features_csv"]
    metadata_path = ROOT / config["paths"]["tracks_csv"]
    manifest_path = ROOT / config["paths"]["feature_manifest"]
    cache_path = ROOT / "results/metrics/representation_fit_checkpoint.joblib"
    selected_path = ROOT / "results/phase5b_svm/metrics/selected_configs.json"
    bundle_path = ROOT / "models/phase5b_svm/phase5b_svm__mutual_information.joblib"
    for path in (feature_path, metadata_path, manifest_path, cache_path,
                 selected_path, bundle_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 8E input is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("feature_columns") != list(FEATURE_COLUMNS) or len(FEATURE_COLUMNS) != 74:
        raise ValueError("Feature manifest does not match the 74-feature schema")
    # Only track IDs and split assignments are loaded from the feature table.
    id_split = pd.read_csv(feature_path, usecols=["track_id", "split"])
    train_ids = id_split.loc[id_split.split.eq("training"), "track_id"].astype(int).tolist()
    validation_ids = id_split.loc[id_split.split.eq("validation"), "track_id"].astype(int).tolist()
    if len(train_ids) != 13_510 or len(validation_ids) != 1_704:
        raise ValueError("Official training/validation row counts differ from the saved experiment")
    if id_split.track_id.duplicated().any():
        raise ValueError("Feature table contains duplicate track IDs")

    cache = joblib.load(cache_path)
    reps = cache["representations"]
    info = cache["representation_info"][REPRESENTATION]
    x_train, x_validation, columns, transformer = reps[REPRESENTATION]
    columns = list(columns)
    if transformer is not None:
        raise ValueError("MI cache unexpectedly requires an additional transform")
    if len(columns) != 37 or len(set(columns)) != 37 or not set(columns).issubset(FEATURE_COLUMNS):
        raise ValueError("Saved MI feature schema/order is invalid")
    if info.get("selected_features") != columns:
        raise ValueError("MI representation metadata order differs from cached matrix order")
    if x_train.shape != (len(train_ids), 37) or x_validation.shape != (len(validation_ids), 37):
        raise ValueError("Cached MI representation dimensions do not match train/validation splits")
    if cache.get("training_ids") != train_ids or cache.get("validation_ids") != validation_ids:
        raise ValueError("Cached MI rows do not match official split ID order")

    labels = _labels_for_splits(metadata_path, config["project"]["dataset_subset"],
                                {"training", "validation"})
    if not set(train_ids + validation_ids).issubset(labels):
        raise ValueError("A successful train/validation feature row has no official genre label")
    y_train = np.asarray([labels[track_id] for track_id in train_ids], dtype=object)
    y_validation = np.asarray([labels[track_id] for track_id in validation_ids], dtype=object)
    class_order = list(config["project"]["class_order"])
    if set(y_train) != set(class_order) or set(y_validation) != set(class_order):
        raise ValueError("Train/validation labels do not cover the configured 16 genres")

    bundle = joblib.load(bundle_path)
    if bundle.get("representation") != REPRESENTATION or bundle.get("selected_columns") != columns:
        raise ValueError("Saved Phase 5B bundle feature order does not match cached MI order")
    if bundle.get("input_dimension") != 37 or bundle.get("training_rows") != len(train_ids):
        raise ValueError("Saved baseline bundle dimensions or training rows are incompatible")
    estimator = bundle.get("model")
    svc = estimator.named_steps.get("svc")
    scaler = estimator.named_steps.get("standardscaler")
    if svc is None or scaler is None or scaler.n_samples_seen_ != len(train_ids):
        raise ValueError("Baseline bundle does not contain a training-fitted StandardScaler/SVC")
    if (svc.kernel != "rbf" or svc.C != 10.0 or svc.gamma != "scale" or
            svc.class_weight is not None or svc.probability is not False):
        raise ValueError("Saved Phase 5B baseline does not match requested SVC settings")

    selection = json.loads(selected_path.read_text(encoding="utf-8"))
    saved = selection["selected"][REPRESENTATION]
    if saved["selected_hyperparameters"] != PARAMS:
        raise ValueError("Saved validation selection is not MI SVC C=10, gamma=scale")
    if saved["validation_metrics"]["macro_f1"] != 0.3828451379993131:
        raise ValueError("Saved baseline validation macro-F1 differs from the supplied baseline")
    return x_train, y_train, x_validation, y_validation, columns, class_order, saved


def _write_csv(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def run() -> None:
    if OUTPUT.exists() or MODEL_OUTPUT.exists():
        raise FileExistsError("Phase 8E output/model path already exists; refusing overwrite")
    before = protected_hashes()
    x_train, y_train, x_validation, y_validation, columns, class_order, saved = load_train_validation()

    # Independently reproduce the Phase 5B baseline using only training and validation.
    base_model, base_record = fit_candidate(
        x_train, y_train, x_validation, y_validation, class_order, PARAMS,
        pipeline_factory=build_conventional_svc)
    for metric_name in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if not np.isclose(base_record["validation_metrics"][metric_name],
                          saved["validation_metrics"][metric_name], rtol=0, atol=1e-12):
            raise RuntimeError(f"Baseline reproduction differs for {metric_name}")

    weighted_model, weighted_record = fit_candidate(
        x_train, y_train, x_validation, y_validation, class_order, PARAMS,
        pipeline_factory=build_balanced_svc)
    weighted_svc = weighted_model.named_steps["svc"]
    base_svc = base_model.named_steps["svc"]
    if (weighted_svc.class_weight != "balanced" or
            any(getattr(weighted_svc, key) != getattr(base_svc, key)
                for key in ("kernel", "C", "gamma", "probability", "decision_function_shape"))):
        raise RuntimeError("Class-balanced model differs from baseline in more than class_weight")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=False)
    metrics_dir = OUTPUT / "metrics"
    metrics_dir.mkdir()
    models_path = MODEL_OUTPUT / "phase8e_mi_svc_balanced.joblib"
    save_new_bundle(models_path, {
        "schema_version": 1, "representation": REPRESENTATION,
        "feature_columns": list(FEATURE_COLUMNS), "selected_columns": columns,
        "labels": class_order, "input_dimension": 37, "training_rows": len(y_train),
        "validation_rows": len(y_validation), "model": weighted_model,
        "model_parameters": {**PARAMS, "kernel": "rbf", "class_weight": "balanced",
                              "probability": False},
        "scaler_fit_rows": len(y_train),
        "note": "Validation-only class-weight experiment; no test data used.",
    })

    records = {"baseline_unweighted": base_record, "class_weight_balanced": weighted_record}
    (metrics_dir / "validation_metrics.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    summary_rows = []
    per_class_rows = []
    confusion_rows = []
    predicted_rows = []
    for name, record in records.items():
        metric = record["validation_metrics"]
        summary_rows.append({"model": name, "accuracy": metric["accuracy"],
                             "macro_precision": metric["macro_precision"],
                             "macro_recall": metric["macro_recall"], "macro_f1": metric["macro_f1"],
                             "training_seconds": record["training_seconds"],
                             "inference_seconds": metric["inference_seconds"],
                             "training_rows": record["training_rows"],
                             "validation_rows": record["validation_rows"], "feature_count": 37})
        matrix = np.asarray(metric["confusion_matrix"], dtype=int)
        for row_i, genre in enumerate(class_order):
            detail = metric["per_class"][genre]
            per_class_rows.append({"model": name, "genre": genre,
                                   "precision": detail["precision"], "recall": detail["recall"],
                                   "f1": detail["f1-score"], "support": int(detail["support"])})
            confusion_rows.append({"model": name, "true_genre": genre,
                                   **{f"predicted_{col}": int(matrix[row_i, col_i])
                                      for col_i, col in enumerate(class_order)}})
            predicted_rows.append({"model": name, "genre": genre,
                                   "predicted_count": int(matrix[:, row_i].sum())})
    _write_csv(metrics_dir / "summary.csv", summary_rows)
    _write_csv(metrics_dir / "per_genre_metrics.csv", per_class_rows)
    _write_csv(metrics_dir / "confusion_matrices.csv", confusion_rows)
    _write_csv(metrics_dir / "predicted_class_counts.csv", predicted_rows)
    (OUTPUT / "run_manifest.json").write_text(json.dumps({
        "phase": "8E", "evaluation_split": "validation", "training_rows": len(y_train),
        "validation_rows": len(y_validation), "feature_count": 37,
        "feature_order": columns, "class_order": class_order, "parameters": PARAMS,
        "baseline_class_weight": None, "comparison_class_weight": "balanced",
        "baseline_validation_macro_f1_saved": saved["validation_metrics"]["macro_f1"],
        "test_data_accessed": False,
    }, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    after = protected_hashes()
    if before != after:
        raise RuntimeError("Protected Phase 5/5B artifacts changed during Phase 8E run")
    (OUTPUT / "protected_artifact_hashes.json").write_text(json.dumps({
        "before": before, "after": after, "unchanged": True,
    }, indent=2), encoding="utf-8")
    print(f"Baseline validation macro-F1: {base_record['validation_metrics']['macro_f1']:.12f}")
    print(f"Balanced validation macro-F1: {weighted_record['validation_metrics']['macro_f1']:.12f}")
    print(f"Saved validation-only results to {OUTPUT}")


if __name__ == "__main__":
    run()
