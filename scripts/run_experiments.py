"""Run leakage-safe FMA Medium representation and classifier comparisons.

All 16 configurations are selected from training/validation data and saved before
the first test prediction. Test predictions are checkpointed exactly once per
configuration; an interrupted, ambiguous test evaluation is never repeated.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
import hashlib
import json
import logging
import sys
import time

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loading import load_track_metadata
from src.dimensionality_reduction import PCATransformer
from src.evaluation import evaluate_predictions, predict_timed
from src.feature_extraction import FEATURE_COLUMNS, feature_config
from src.feature_selection import (candidate_counts, mutual_information_ranking,
                                   random_forest_ranking, select_top_features)
from src.model_training import train_mlp, train_sklearn
from src.plotting import (plot_confusion, plot_feature_importance,
                          plot_metric_comparison, plot_pca_variance)
from src.prediction import load_predictor, save_predictor
from src.utils import atomic_json, configure_logging, load_config, project_path, set_seed, stable_hash

LOG = logging.getLogger(__name__)
MODELS = ("logistic_regression", "svm", "random_forest", "mlp")
REPRESENTATIONS = ("full", "mutual_information", "random_forest", "pca_95")
CHECKPOINT_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_inputs(config: dict, feature_path: Path, metadata: pd.DataFrame,
                   features: pd.DataFrame, failures: pd.DataFrame, manifest: dict) -> None:
    expected_columns = ["track_id", "genre", "split", *FEATURE_COLUMNS]
    if list(features.columns) != expected_columns:
        raise ValueError("Feature CSV columns/order do not match the expected 74-feature schema")
    if len(features) != 16_986 or features.track_id.nunique() != len(features):
        raise ValueError(f"Expected 16,986 unique feature rows; got {len(features)} rows and "
                         f"{features.track_id.nunique()} unique IDs")
    matrix = features[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature CSV contains NaN or infinite values")
    if features[["track_id", "genre", "split"]].isna().any().any():
        raise ValueError("Feature CSV contains missing ID, genre, or split metadata")
    if failures.track_id.duplicated().any() or failures.reason.isna().any() or len(failures) != 14:
        raise ValueError("Failure log is malformed, duplicated, or differs from the verified 14 failures")

    official_ids = set(metadata.index.astype(int))
    feature_ids = set(features.track_id.astype(int))
    failure_ids = set(failures.track_id.astype(int))
    if feature_ids & failure_ids or feature_ids | failure_ids != official_ids:
        raise ValueError("Feature and failure IDs do not partition official FMA Medium metadata")
    genre_mismatch = features.genre.astype(str).ne(features.track_id.map(metadata.genre).astype(str))
    split_mismatch = features.split.astype(str).ne(features.track_id.map(metadata.split).astype(str))
    if genre_mismatch.any() or split_mismatch.any():
        raise ValueError("Feature genres or official split assignments disagree with metadata")
    if set(features.split.unique()) != {"training", "validation", "test"}:
        raise ValueError("Feature data does not contain all three official splits")

    expected_signature = stable_hash({"feature_config": feature_config(config), "columns": FEATURE_COLUMNS})
    if manifest.get("signature") != expected_signature:
        raise ValueError("Feature manifest signature does not match the configured schema")
    if (manifest.get("successful_tracks") != len(features)
            or manifest.get("failed_tracks") != len(failures)
            or manifest.get("feature_columns") != list(FEATURE_COLUMNS)):
        raise ValueError("Feature manifest counts/schema do not agree with saved CSV files")
    LOG.info("Input audit passed: %d rows, %d finite features; official split counts=%s; failures=%d",
             len(features), len(FEATURE_COLUMNS), features.split.value_counts().to_dict(), len(failures))


def _fit_representations(train: pd.DataFrame, validation: pd.DataFrame, config: dict):
    names = list(FEATURE_COLUMNS)
    Xtr, Xva = train[names], validation[names]
    ytr, yva = train.genre, validation.genre
    seed = int(config["project"]["seed"])
    start = time.perf_counter()
    mi_rank = mutual_information_ranking(Xtr, ytr, seed)
    mi_rank_seconds = time.perf_counter() - start
    start = time.perf_counter()
    rf_rank = random_forest_ranking(Xtr, ytr, seed, config["selection"]["random_forest_trees"])
    rf_rank_seconds = time.perf_counter() - start

    representations = {"full": (Xtr, Xva, names, None)}
    # The full representation has the same metadata contract as the selected
    # representations, except that its feature list is the complete schema.
    selection_info = {"full": {"selected_count": len(names),
                                "selected_features": list(names),
                                "selection_method": "all configured features"}}
    for rep_name, ranking, counts in (
        ("mutual_information", mi_rank,
         candidate_counts(len(names), config["selection"]["mutual_information_fractions"])),
        ("random_forest", rf_rank, config["selection"]["random_forest_counts"]),
    ):
        best = None
        for k in sorted({max(1, min(len(names), int(value))) for value in counts}):
            selected = select_top_features(ranking, k)
            estimator, _ = train_sklearn("logistic_regression", Xtr[selected], ytr, config, seed)
            score = f1_score(yva, estimator.predict(Xva[selected]), average="macro", zero_division=0)
            LOG.info("Representation selection %s k=%d validation macro-F1=%.5f", rep_name, k, score)
            if best is None or (score, -k) > (best[0], -best[1]):
                best = (score, k, selected)
        score, k, selected = best
        representations[rep_name] = (Xtr[selected], Xva[selected], selected, None)
        selection_info[rep_name] = {"validation_macro_f1": float(score), "selected_count": int(k),
                                    "selected_features": list(selected)}

    pca = PCATransformer(float(config["pca"]["variance_thresholds"][0])).fit(Xtr)
    representations["pca_95"] = (pca.transform(Xtr), pca.transform(Xva), names, pca)
    selection_info["pca_95"] = {"retained_components": int(pca.pca.n_components_),
                                 "explained_variance": float(pca.pca.explained_variance_ratio_.sum()),
                                 "original_features": len(names)}
    return representations, {"mutual_information": mi_rank, "random_forest": rf_rank}, selection_info, {
        "mutual_information_ranking_seconds": mi_rank_seconds,
        "random_forest_ranking_seconds": rf_rank_seconds,
    }


def _candidate_configs(model_name: str, config: dict):
    base = deepcopy(config)
    search = config["models"]["validation_search"][model_name]
    if model_name == "logistic_regression":
        params = [{"C": value} for value in search["C"]]
    elif model_name == "svm":
        params = [{"C": c, "gamma": gamma} for c, gamma in product(search["C"], search["gamma"])]
    elif model_name == "random_forest":
        params = [{"n_estimators": n, "max_depth": depth}
                  for n, depth in product(search["n_estimators"], search["max_depth"])]
    elif model_name == "mlp":
        params = [{"hidden_layers": layer, "learning_rate": lr}
                  for layer, lr in product(search["hidden_layers"], search["learning_rate"])]
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    candidates = []
    for one in params:
        candidate = deepcopy(base)
        candidate["models"][model_name].update(one)
        candidates.append(candidate)
    return candidates, params


def _select_model_on_validation(model_name, xtr, ytr, xva, yva, config, seed):
    candidates, params_list = _candidate_configs(model_name, config)
    best = None
    for index, (candidate_config, params) in enumerate(zip(candidates, params_list), start=1):
        LOG.info("Tuning %s candidate %d/%d params=%s", model_name, index, len(candidates), params)
        if model_name == "mlp":
            fitted = train_mlp(xtr, ytr, xva, yva, candidate_config, seed)
            train_seconds = fitted.training_seconds
        else:
            fitted, train_seconds = train_sklearn(model_name, xtr, ytr, candidate_config, seed)
        val_pred, val_infer_seconds = predict_timed(fitted, xva)
        score = f1_score(yva, val_pred, average="macro", zero_division=0)
        LOG.info("Candidate validation macro-F1=%.5f training_seconds=%.2f", score, train_seconds)
        if best is None or score > best[0]:
            best = (score, fitted, train_seconds, val_pred, val_infer_seconds, params)
    if best is None:
        raise RuntimeError(f"No candidates configured for {model_name}")
    return best[1:]


def _predict_from_bundle(bundle: dict, full_features: pd.DataFrame) -> np.ndarray:
    selected = bundle["selected_columns"]
    values = full_features.loc[:, selected]
    transformer = bundle.get("transformer")
    if transformer is not None:
        values = transformer.transform(values)
    return np.asarray(bundle["model"].predict(values))


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _validate_representation_cache(cache: dict, signature: str, train: pd.DataFrame,
                                   validation: pd.DataFrame) -> None:
    if cache.get("run_signature") != signature:
        raise ValueError("Representation cache signature does not match current features/configuration")
    if cache.get("training_ids") != train.track_id.astype(int).tolist():
        raise ValueError("Representation cache training IDs differ from the current official training split")
    if cache.get("validation_ids") != validation.track_id.astype(int).tolist():
        raise ValueError("Representation cache validation IDs differ from the current official validation split")
    if cache.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("Representation cache feature schema/order is incompatible")
    representations = cache.get("representations", {})
    info = cache.get("representation_info", {})
    if set(representations) != set(REPRESENTATIONS) or set(info) != set(REPRESENTATIONS):
        raise ValueError("Representation cache does not contain all four representations")
    for name in REPRESENTATIONS:
        xtr, xva, columns, transformer = representations[name]
        if xtr.shape[0] != len(train) or xva.shape[0] != len(validation):
            raise ValueError(f"Cached {name} row counts do not match current training/validation partitions")
        if name in ("full", "mutual_information", "random_forest"):
            if list(columns) != info[name].get("selected_features"):
                raise ValueError(f"Cached {name} selected feature names/order are inconsistent")
        if name == "full" and (xtr.shape[1] != len(FEATURE_COLUMNS) or transformer is not None):
            raise ValueError("Cached full representation has an invalid shape or transformer")
        if name == "pca_95":
            if transformer is None or list(transformer.feature_names_in_ or []) != list(FEATURE_COLUMNS):
                raise ValueError("Cached PCA transformer does not match the training feature schema")
            if xtr.shape[1] != info[name].get("retained_components"):
                raise ValueError("Cached PCA component count disagrees with its metadata")
    for name in ("mutual_information", "random_forest"):
        if list(cache["rankings"][name].index) == [] or set(cache["rankings"][name].index) != set(FEATURE_COLUMNS):
            raise ValueError(f"Cached {name} rankings do not cover all training features")


def _validate_model_bundle(bundle: dict, selected: dict, representation: str,
                           class_order: list[str], audio_config: dict) -> None:
    if bundle.get("representation") != representation:
        raise ValueError(f"Saved {representation} model bundle has the wrong representation")
    if bundle.get("selected_columns") != selected.get("selected_features"):
        raise ValueError(f"Saved {representation} model feature order differs from its checkpoint")
    if bundle.get("labels") != class_order or bundle.get("audio_config") != audio_config:
        raise ValueError("Saved model labels/audio configuration differ from current experiment")
    if bundle.get("model") is None:
        raise ValueError("Checkpointed model artifact contains no fitted estimator")


def main():
    configure_logging()
    config = load_config()
    seed = int(config["project"]["seed"])
    set_seed(seed)
    root = Path(config["_project_root"])
    feature_path = project_path(config, config["paths"]["features_csv"])
    failures_path = project_path(config, config["paths"]["failures_csv"])
    manifest_path = project_path(config, config["paths"]["feature_manifest"])
    if not all(path.is_file() for path in (feature_path, failures_path, manifest_path)):
        raise FileNotFoundError("Required feature CSV, failure CSV, or feature manifest is missing")
    features = pd.read_csv(feature_path)
    failures = pd.read_csv(failures_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = load_track_metadata(project_path(config, config["paths"]["tracks_csv"]),
                                   project_path(config, config["paths"]["audio_dir"]),
                                   subset=config["project"]["dataset_subset"])
    _verify_inputs(config, feature_path, metadata, features, failures, manifest)

    reports_dir = project_path(config, config["paths"]["results_dir"]) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    log_path = reports_dir / "phase5_experiments.log"
    if not any(isinstance(handler, logging.FileHandler)
               and Path(handler.baseFilename) == log_path.resolve()
               for handler in root_logger.handlers):
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root_logger.addHandler(file_handler)

    # Only successful feature records are modelable; their saved official split
    # labels remain authoritative. No test values are used in representation fitting.
    train = features.loc[features.split.eq("training")].reset_index(drop=True)
    validation = features.loc[features.split.eq("validation")].reset_index(drop=True)
    test = features.loc[features.split.eq("test")].reset_index(drop=True)
    class_order = list(config["project"]["class_order"])
    if set(train.genre) != set(class_order) or set(validation.genre) != set(class_order) or set(test.genre) != set(class_order):
        raise ValueError("A successful official partition is missing one or more configured genre classes")
    models_dir = project_path(config, config["paths"]["models_dir"])
    metrics_dir = project_path(config, config["paths"]["results_dir"]) / "metrics"
    figures_dir = project_path(config, config["paths"]["results_dir"]) / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    state_path = metrics_dir / "phase5_checkpoint.json"
    data_hash = _sha256(feature_path)
    run_signature = stable_hash({
        "checkpoint_version": CHECKPOINT_VERSION, "feature_sha256": data_hash,
        "manifest_signature": manifest["signature"], "config": config,
        "class_order": class_order,
    })
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_signature") != run_signature:
            raise ValueError("Existing Phase 5 checkpoint does not match current data/configuration")
    else:
        state = {"checkpoint_version": CHECKPOINT_VERSION, "run_signature": run_signature,
                 "feature_sha256": data_hash, "stage": "validation_selection", "selected": {},
                 "test_results": {}, "test_in_progress": None,
                 "test_protocol": "one batch prediction per representation/model after all 16 validation selections are finalized"}
        atomic_json(state_path, state)
    if state.get("test_in_progress"):
        raise RuntimeError("An earlier test evaluation was marked in progress when interrupted. "
                           "To avoid repeating a test evaluation, this run has stopped for review.")

    # Persist representation rankings, validation-chosen feature subsets, and
    # train-fitted PCA so later interruptions do not repeat those steps.
    representation_cache_path = metrics_dir / "representation_fit_checkpoint.joblib"
    if representation_cache_path.exists():
        representation_cache = joblib.load(representation_cache_path)
        _validate_representation_cache(representation_cache, run_signature, train, validation)
        representations = representation_cache["representations"]
        rankings = representation_cache["rankings"]
        representation_info = representation_cache["representation_info"]
        ranking_times = representation_cache["ranking_times_seconds"]
        LOG.info("Reused validated train-only representation cache at %s", representation_cache_path)
    else:
        representations, rankings, representation_info, ranking_times = _fit_representations(
            train, validation, config)
        representation_cache = {
            "run_signature": run_signature,
            "training_ids": train.track_id.astype(int).tolist(),
            "validation_ids": validation.track_id.astype(int).tolist(),
            "feature_columns": list(FEATURE_COLUMNS),
            "representations": representations,
            "rankings": rankings,
            "representation_info": representation_info,
            "ranking_times_seconds": ranking_times,
        }
        temp_cache = representation_cache_path.with_suffix(".joblib.tmp")
        joblib.dump(representation_cache, temp_cache)
        temp_cache.replace(representation_cache_path)
        _validate_representation_cache(representation_cache, run_signature, train, validation)
        LOG.info("Saved train-only representation cache at %s", representation_cache_path)

    # Persist selected model bundles one at a time. If validation selection is
    # interrupted, only missing combinations are fitted on a restart.
    for rep_name in REPRESENTATIONS:
        xtr, xva, _, transformer = representations[rep_name]
        ytr, yva = train.genre, validation.genre
        for model_name in MODELS:
            key = f"{rep_name}__{model_name}"
            artifact_path = models_dir / f"{key}.joblib"
            if key in state["selected"]:
                if not artifact_path.is_file():
                    raise FileNotFoundError(f"Checkpointed validation-selected model is missing: {artifact_path}")
                bundle = load_predictor(artifact_path)
                _validate_model_bundle(bundle, state["selected"][key], rep_name,
                                       class_order, config["audio"])
                continue
            fitted, train_seconds, val_pred, val_infer_seconds, chosen_params = _select_model_on_validation(
                model_name, xtr, ytr, xva, yva, config, seed)
            rep_info = representation_info[rep_name]
            selected_columns = list(rep_info["selected_features"]) if rep_name in (
                "mutual_information", "random_forest") else list(FEATURE_COLUMNS)
            save_predictor(artifact_path, fitted, selected_columns, class_order,
                           config["audio"], rep_name, transformer)
            val_metrics = evaluate_predictions(yva, val_pred, class_order,
                                               val_infer_seconds, len(val_pred))
            state["selected"][key] = {
                "representation": rep_name, "model": model_name,
                "original_features": len(FEATURE_COLUMNS), "input_features": int(xtr.shape[1]),
                "selected_hyperparameters": chosen_params,
                "training_seconds": float(train_seconds),
                "validation_inference_seconds": float(val_infer_seconds),
                "best_epoch": getattr(fitted, "best_epoch", None),
                "device": getattr(fitted, "device", None),
                "pca_retained_components": rep_info.get("retained_components"),
                "pca_explained_variance": rep_info.get("explained_variance"),
                "selected_features": selected_columns,
                "validation_metrics": val_metrics,
                "artifact": str(artifact_path),
            }
            atomic_json(state_path, state)
            LOG.info("Saved validation-selected %s (%s); macro-F1=%.5f",
                     key, chosen_params, val_metrics["macro_f1"])

    expected_keys = {f"{rep}__{model}" for rep in REPRESENTATIONS for model in MODELS}
    if set(state["selected"]) != expected_keys:
        raise RuntimeError("Not all 16 validation-selected configurations are checkpointed")
    # Validate every final candidate bundle before opening the test-evaluation
    # stage. This prevents partial/corrupt artifacts from reaching test data.
    for key in sorted(expected_keys):
        selected = state["selected"][key]
        artifact = Path(selected["artifact"])
        if not artifact.is_file():
            raise FileNotFoundError(f"Final validation-selected model is missing: {artifact}")
        bundle = load_predictor(artifact)
        _validate_model_bundle(bundle, selected, selected["representation"],
                               class_order, config["audio"])
    if state["stage"] == "validation_selection":
        state["stage"] = "test_evaluation_ready"
        state["representations"] = representation_info
        state["ranking_times_seconds"] = ranking_times
        atomic_json(state_path, state)
    if state["stage"] not in ("test_evaluation_ready", "test_evaluation", "complete"):
        raise RuntimeError(f"Unexpected experiment checkpoint stage: {state['stage']}")

    # First test access is after every selected hyperparameter, feature subset,
    # and PCA transformer has been persisted above.
    if state["stage"] != "complete":
        state["stage"] = "test_evaluation"
        atomic_json(state_path, state)
    for rep_name in REPRESENTATIONS:
        for model_name in MODELS:
            key = f"{rep_name}__{model_name}"
            if key in state["test_results"]:
                continue
            if key in (state.get("test_in_progress") or []):
                raise RuntimeError(f"Test evaluation for {key} has ambiguous interrupted state; refusing to repeat")
            state["test_in_progress"] = [key]
            atomic_json(state_path, state)
            bundle = load_predictor(state["selected"][key]["artifact"])
            _sync_cuda()
            start = time.perf_counter()
            test_pred = _predict_from_bundle(bundle, test[FEATURE_COLUMNS])
            _sync_cuda()
            inference_seconds = time.perf_counter() - start
            test_metrics = evaluate_predictions(test.genre, test_pred, class_order,
                                               inference_seconds, len(test_pred))
            state["test_results"][key] = test_metrics
            state["test_in_progress"] = None
            atomic_json(state_path, state)
            LOG.info("One-time test evaluation complete for %s: accuracy=%.5f macro-F1=%.5f",
                     key, test_metrics["accuracy"], test_metrics["macro_f1"])

    results = []
    for key in sorted(expected_keys):
        selected = state["selected"][key]
        results.append({**{k: v for k, v in selected.items()
                           if k not in ("validation_metrics", "selected_features", "artifact")},
                        "representation": selected["representation"], "model": selected["model"],
                        "selected_feature_count": selected["input_features"],
                        "selected_features": selected["selected_features"],
                        "partition": "validation", **selected["validation_metrics"]})
        results.append({**{k: v for k, v in selected.items()
                           if k not in ("validation_metrics", "selected_features", "artifact")},
                        "representation": selected["representation"], "model": selected["model"],
                        "selected_feature_count": selected["input_features"],
                        "selected_features": selected["selected_features"],
                        "partition": "test", **state["test_results"][key]})
    atomic_json(metrics_dir / "experiment_results.json", results)
    flat = pd.json_normalize(results, sep=".")
    flat.drop(columns=[c for c in flat.columns if c.startswith("per_class.")
                       or c == "confusion_matrix"], errors="ignore").to_csv(
                           metrics_dir / "experiment_results.csv", index=False)
    per_class_rows = []
    for row in results:
        for genre, metrics in row["per_class"].items():
            per_class_rows.append({"representation": row["representation"], "model": row["model"],
                                   "partition": row["partition"], "genre": genre, **metrics})
    pd.DataFrame(per_class_rows).to_csv(metrics_dir / "per_class_metrics.csv", index=False)
    atomic_json(metrics_dir / "representation_selection.json", {
        "representations": representation_info,
        "ranking_times_seconds": ranking_times,
        "selected_features": {name: state["selected"][f"{name}__logistic_regression"]["selected_features"]
                              for name in ("mutual_information", "random_forest")},
        "test_evaluations": len(state["test_results"]),
        "test_set_access_policy": state["test_protocol"],
        "inference_timing_method": "wall-clock batch prediction on the held-out successful test rows; includes saved PCA transform when applicable and model preprocessing, excludes audio decoding/feature extraction; CUDA synchronized before and after timing",
    })
    for name, ranking in rankings.items():
        ranking.rename_axis("feature").reset_index().to_csv(metrics_dir / f"{name}_ranking.csv", index=False)
    pca = representations["pca_95"][3].pca
    pd.DataFrame({"component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
                  "explained_variance_ratio": pca.explained_variance_ratio_,
                  "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_)
                  }).to_csv(metrics_dir / "pca_explained_variance.csv", index=False)
    plot_feature_importance(rankings["random_forest"], figures_dir / "random_forest_feature_importance.png")
    plot_pca_variance(pca, figures_dir / "pca_cumulative_variance.png")
    plot_metric_comparison(flat, "accuracy", figures_dir / "test_accuracy_comparison.png")
    plot_metric_comparison(flat, "macro_f1", figures_dir / "test_macro_f1_comparison.png")
    for row in results:
        if row["partition"] == "test":
            plot_confusion(row["confusion_matrix"], class_order,
                           figures_dir / f"confusion_{row['representation']}__{row['model']}.png",
                           f"{row['representation']} / {row['model']} test confusion matrix")
    state["stage"] = "complete"
    state["completed_at_epoch_seconds"] = time.time()
    atomic_json(state_path, state)
    LOG.info("Phase 5 complete. Results written to %s", metrics_dir)


if __name__ == "__main__":
    main()
