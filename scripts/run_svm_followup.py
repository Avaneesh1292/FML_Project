"""Run the isolated Phase 5B full-training-set RBF SVC experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation import evaluate_predictions
from src.feature_extraction import FEATURE_COLUMNS
from src.svm_followup import (assert_fresh_report_path, atomic_json,
                              build_conventional_svc, candidate_grid,
                              fit_candidate, load_or_create_checkpoint,
                              save_new_bundle, validate_representation_cache)
from src.utils import stable_hash

LOG = logging.getLogger("phase5b_svm")
REPRESENTATIONS = ("full", "mutual_information", "random_forest", "pca_95")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    os.replace(temp, path)


def _protected_hashes() -> dict[str, str]:
    """Hash Phase 5 inputs/source/results without including Phase 5B outputs."""
    paths: set[Path] = set()
    for folder in ("src", "scripts", "tests"):
        for p in (ROOT / folder).rglob("*.py"):
            if p.relative_to(ROOT).as_posix() in {
                "src/svm_followup.py", "scripts/run_svm_followup.py",
                "tests/test_svm_followup.py",
            }:
                continue
            paths.add(p)
    paths.add(ROOT / "config/config.yaml")
    for folder in ("results/metrics", "results/figures", "models"):
        for p in (ROOT / folder).rglob("*"):
            if p.is_file() and "phase5b_svm" not in p.relative_to(ROOT).as_posix().lower():
                paths.add(p)
    for relative in (
        "data/features/fma_medium_features.csv",
        "data/features/feature_failures.csv",
        "data/features/feature_manifest.json",
        "fma_metadata/tracks.csv",
    ):
        paths.add(ROOT / relative)
    return {p.relative_to(ROOT).as_posix(): _sha256(p) for p in sorted(paths)
            if p.is_file()}


def _verify_protected_hashes(results_dir: Path) -> None:
    audit_dir = results_dir / "audit"
    before_path = audit_dir / "phase5_baseline_sha256.json"
    current = _protected_hashes()
    if before_path.exists():
        before = json.loads(before_path.read_text(encoding="utf-8"))
        if before != current:
            changed = sorted(set(before) | set(current))
            changed = [name for name in changed if before.get(name) != current.get(name)]
            raise RuntimeError(f"Protected Phase 5 files changed; refusing to continue: {changed}")
    else:
        atomic_json(before_path, current)


def _labels_for_splits(metadata_csv: Path, subset: str,
                       wanted_splits: set[str]) -> dict[int, str]:
    """Read labels only for the requested official splits.

    The row parser checks subset/split before retrieving the top-level genre
    field. In particular, the tuning call requests training and validation only.
    """
    labels: dict[int, str] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        top = next(reader); second = next(reader)
        headers = list(zip(top, second))
        try:
            subset_i = headers.index(("set", "subset"))
            split_i = headers.index(("set", "split"))
            genre_i = headers.index(("track", "genre_top"))
        except ValueError as exc:
            raise ValueError("FMA tracks.csv lacks required subset/split/genre_top columns") from exc
        for row in reader:
            if not row:
                continue
            if row[subset_i] != subset or row[split_i] not in wanted_splits:
                continue
            track_id = int(row[0])
            genre = row[genre_i]
            if not genre:
                raise ValueError(f"Official top-level genre missing for track {track_id}")
            if track_id in labels:
                raise ValueError(f"Duplicate official track ID {track_id}")
            labels[track_id] = genre
    return labels


def _load_preflight(base_config: dict, phase5b_config: dict):
    paths = base_config["paths"]
    feature_path = ROOT / paths["features_csv"]
    metadata_path = ROOT / paths["tracks_csv"]
    manifest_path = ROOT / paths["feature_manifest"]
    phase5_checkpoint_path = ROOT / "results/metrics/phase5_checkpoint.json"
    cache_path = ROOT / "results/metrics/representation_fit_checkpoint.joblib"
    for path in (feature_path, metadata_path, manifest_path,
                 phase5_checkpoint_path, cache_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 5B input is missing: {path}")

    phase5_state = json.loads(phase5_checkpoint_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_hash = _sha256(feature_path)
    if phase5_state.get("stage") != "complete" or phase5_state.get("test_in_progress"):
        raise ValueError("Phase 5 checkpoint is not complete or has ambiguous test state")
    if phase5_state.get("feature_sha256") != feature_hash:
        raise ValueError("Feature CSV hash differs from completed Phase 5 checkpoint")
    if manifest.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("Feature manifest schema differs from the configured 74-feature schema")

    # Load IDs/split metadata only. Test labels are deliberately absent here.
    id_split = pd.read_csv(feature_path, usecols=["track_id", "split"])
    if len(id_split) != 16_986 or not id_split.track_id.is_unique:
        raise ValueError("Feature IDs are duplicated or successful-row count differs from Phase 5")
    if id_split.split.value_counts().to_dict() != {
        "training": 13_510, "validation": 1_704, "test": 1_772
    }:
        raise ValueError("Saved feature split counts differ from the verified Phase 5 counts")
    train_ids = id_split.loc[id_split.split.eq("training"), "track_id"].astype(int).tolist()
    validation_ids = id_split.loc[id_split.split.eq("validation"), "track_id"].astype(int).tolist()

    cache = joblib.load(cache_path)
    validate_representation_cache(
        cache, run_signature=phase5_state["run_signature"],
        training_ids=train_ids, validation_ids=validation_ids,
        feature_columns=list(FEATURE_COLUMNS), expected_train_rows=len(train_ids),
        expected_validation_rows=len(validation_ids),
    )
    pca_transformer = cache["representations"]["pca_95"][3]
    expected_variance = float(phase5_state["representations"]["pca_95"]["explained_variance"])
    if not np.isclose(pca_transformer.pca.explained_variance_ratio_.sum(),
                      expected_variance, rtol=1e-10, atol=1e-12):
        raise ValueError("Cached PCA explained variance differs from Phase 5 metadata")

    # Only train/validation labels are materialized for candidate fitting.
    labels = _labels_for_splits(metadata_path, base_config["project"]["dataset_subset"],
                                {"training", "validation"})
    successful_trainval = set(train_ids) | set(validation_ids)
    # Official metadata also contains the 12 train/validation tracks excluded
    # by Phase 4 feature-extraction failures. Require every cached ID to have
    # its official label, while allowing those explicitly failed IDs to remain
    # absent from the feature cache.
    if not successful_trainval.issubset(labels):
        raise ValueError("Some successful train/validation feature IDs lack official labels")
    if any(phase5_state["selected"][f"{rep}__svm"]["model"] != "svm"
           for rep in REPRESENTATIONS):
        raise ValueError("Phase 5 checkpoint does not contain all four original SVM baselines")

    class_order = list(base_config["project"]["class_order"])
    if set(labels.values()) != set(class_order):
        raise ValueError("Train/validation metadata labels differ from configured classes")
    y_train = np.asarray([labels[tid] for tid in train_ids], dtype=object)
    y_validation = np.asarray([labels[tid] for tid in validation_ids], dtype=object)
    x_train = {rep: cache["representations"][rep][0] for rep in REPRESENTATIONS}
    x_validation = {rep: cache["representations"][rep][1] for rep in REPRESENTATIONS}
    columns = {rep: list(cache["representation_info"][rep].get(
        "selected_features", FEATURE_COLUMNS)) for rep in REPRESENTATIONS}
    return (phase5_state, cache, feature_path, metadata_path, feature_hash,
            train_ids, validation_ids, y_train, y_validation,
            x_train, x_validation, columns, class_order)


def _record_key(rep: str, index: int) -> str:
    return f"{rep}__candidate_{index:02d}"


def _candidate_records(state: dict, rep: str) -> dict[int, dict]:
    found = {}
    for row in state["candidate_results"]:
        if row["representation"] == rep:
            index = int(row["candidate_index"])
            if index in found:
                raise ValueError(f"Duplicate checkpointed candidate: {rep} index {index}")
            found[index] = row
    return found


def _make_record(estimator, params, rep, candidate_index, metrics_record):
    val_metrics = metrics_record["validation_metrics"]
    return {
        "representation": rep, "candidate_index": int(candidate_index),
        "C": float(params["C"]), "gamma": params["gamma"],
        "validation_macro_f1": float(val_metrics["macro_f1"]),
        "validation_metrics": val_metrics,
        "training_seconds": float(metrics_record["training_seconds"]),
        "training_rows": int(metrics_record["training_rows"]),
        "validation_rows": int(metrics_record["validation_rows"]),
    }


def _save_validation_outputs(results_dir: Path, state: dict, phase5b_config: dict) -> None:
    metrics_dir = results_dir / "metrics"
    atomic_json(metrics_dir / "validation_grid.json", state["candidate_results"])
    grid_rows = []
    for row in state["candidate_results"]:
        metrics = row["validation_metrics"]
        grid_rows.append({"representation": row["representation"],
                          "candidate_index": row["candidate_index"],
                          "C": row["C"], "gamma": row["gamma"],
                          "validation_accuracy": metrics["accuracy"],
                          "validation_macro_precision": metrics["macro_precision"],
                          "validation_macro_recall": metrics["macro_recall"],
                          "validation_macro_f1": metrics["macro_f1"],
                          "training_seconds": row["training_seconds"],
                          "training_rows": row["training_rows"],
                          "validation_rows": row["validation_rows"]})
    _atomic_csv(metrics_dir / "validation_grid.csv", grid_rows)
    atomic_json(metrics_dir / "selected_configs.json", {
        "selection_metric": "validation macro-F1",
        "search": phase5b_config["search"],
        "selected": state["selected"],
    })


def _check_bundle(path: Path, rep: str, selected: dict, class_order: list[str],
                  feature_hash: str, train_rows: int, input_dim: int):
    if not path.is_file():
        raise FileNotFoundError(f"Selected Phase 5B bundle is missing: {path}")
    bundle = joblib.load(path)
    if bundle.get("phase5b_feature_sha256") != feature_hash:
        raise ValueError(f"Bundle {path.name} is tied to different features")
    if bundle.get("representation") != rep or bundle.get("labels") != class_order:
        raise ValueError(f"Bundle {path.name} has incompatible representation or labels")
    if bundle.get("selected_hyperparameters") != selected["selected_hyperparameters"]:
        raise ValueError(f"Bundle {path.name} params differ from checkpoint")
    model = bundle.get("model")
    if model is None or model.steps[-1][1].kernel != "rbf":
        raise ValueError(f"Bundle {path.name} is not a fitted RBF SVC pipeline")
    scaler = model.named_steps["standardscaler"]
    if int(scaler.n_samples_seen_) != train_rows or model.steps[-1][1].n_features_in_ != input_dim:
        raise ValueError(f"Bundle {path.name} scaler/model fit dimensions do not match training data")
    del bundle


def _test_frame_after_selection(feature_path: Path, metadata_path: Path,
                                feature_hash: str, trainval_ids: set[int],
                                expected_test_ids: list[int], subset: str) -> pd.DataFrame:
    # This function is called only once the validation-complete checkpoint and
    # all four bundles have been persisted and audited.
    current_hash = _sha256(feature_path)
    if current_hash != feature_hash:
        raise ValueError("Feature CSV changed before exploratory test evaluation")
    test_labels = _labels_for_splits(metadata_path, subset, {"test"})
    expected_ids = set(expected_test_ids)
    if not expected_ids.issubset(test_labels) or set(test_labels) & trainval_ids:
        raise ValueError("Some successful test IDs lack official labels or overlap train/validation")
    # Metadata includes the single test track that failed feature extraction;
    # evaluate exactly the successful held-out feature rows.
    test_labels = {track_id: test_labels[track_id] for track_id in expected_test_ids}
    df = pd.read_csv(feature_path, usecols=["track_id", "genre", "split", *FEATURE_COLUMNS])
    test = df.loc[df.split.eq("test")].copy().reset_index(drop=True)
    if test.track_id.astype(int).tolist() != expected_test_ids:
        raise ValueError("Test feature row order differs from the verified split listing")
    if not test.track_id.astype(int).map(test_labels).eq(test.genre).all():
        raise ValueError("Test labels in feature CSV disagree with official metadata")
    return test


def _report_text(state: dict, phase5_checkpoint: dict, base_config: dict,
                 phase5_results_path: Path, phase5b_config: dict) -> str:
    old = pd.read_csv(phase5_results_path)
    old = old[(old.model == "svm") & old.representation.isin(REPRESENTATIONS)]
    old_val = old[old.partition.eq("validation")].set_index("representation")
    old_test = old[old.partition.eq("test")].set_index("representation")
    lines = [
        "# Phase 5B — Conventional RBF SVC follow-up", "",
        "> This follow-up compares the original Phase 5 calibrated SVC baseline with a conventional full-training-set RBF SVC. All Phase 5 test results had already been inspected before this run; Phase 5B test scores are exploratory only. Test metrics were not used for parameter selection.", "",
        "## Method", "",
        "The run reused Phase 5 successful training and validation IDs, official labels, feature order, selected MI/RF subsets, and the saved training-fitted PCA transform. The four representations were: full 74 features, Mutual Information 37 features, Random Forest importance 40 features, and PCA 45 components (95.30% training variance). Preflight verified the representation cache signature against the completed Phase 5 checkpoint, exact train/validation IDs and row counts, feature schema, and that the PCA scaler/PCA were fit on the 13,510 training rows. No representation was refit.", "",
        f"The Phase 5B estimator was `StandardScaler` followed by one `SVC(kernel='rbf', probability=False, class_weight=None)` fitted on all 13,510 training rows. The scaler was part of the pipeline and fit only on training inputs. The prespecified grid was C={phase5b_config['search']['C']} and gamma={phase5b_config['search']['gamma']} (9 candidates per representation, 36 total); parameters were selected independently per representation using validation macro-F1 across all 16 labels.", "",
        "### Difference from original baseline", "",
        "The original Phase 5 SVM wrapped an RBF SVC in `CalibratedClassifierCV(cv=3, method='sigmoid')`. In the saved Phase 5 bundles this produced a three-member calibrated ensemble, with each base SVC trained on a CV fold subset. Phase 5B removes calibration and fits one base SVC on the entire training partition. Both versions use unweighted classes and scaled model inputs. Phase 5B does not provide class probabilities; Phase 5 calibrated baseline does. Because of these training and prediction differences, this is a focused model-implementation comparison, not a perfectly isolated hyperparameter-only comparison.", "",
        "## Validation search and selection", "",
        "| Representation | Phase 5 calibrated SVM val macro-F1 | Phase 5B selected C | gamma | Phase 5B validation accuracy | Macro precision | Macro recall | Macro-F1 | Training rows |", "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rep in REPRESENTATIONS:
        sel = state["selected"][rep]
        metric = sel["validation_metrics"]
        old_score = float(old_val.loc[rep, "macro_f1"])
        lines.append(f"| {rep} | {old_score:.4f} | {sel['selected_hyperparameters']['C']:g} | {sel['selected_hyperparameters']['gamma']} | {metric['accuracy']:.4f} | {metric['macro_precision']:.4f} | {metric['macro_recall']:.4f} | {metric['macro_f1']:.4f} | {sel['training_rows']} |")
    lines += ["", "The validation-selected settings above were fixed and all four bundles saved before exploratory test access. The candidate-level results are in `results/phase5b_svm/metrics/validation_grid.csv` and `.json`.", "", "## Exploratory test comparison", "", "These are exploratory results because the Phase 5 test outcomes were already inspected. They are not used to tune or select any Phase 5B configuration and should not be presented as an unbiased final test estimate.", "", "| Representation | Original Phase 5 calibrated SVM test macro-F1 | Phase 5B exploratory accuracy | Macro precision | Macro recall | Macro-F1 |", "|---|---:|---:|---:|---:|---:|"]
    tests = state["exploratory_test_results"]
    for rep in REPRESENTATIONS:
        metric = tests[rep]
        old_score = float(old_test.loc[rep, "macro_f1"])
        lines.append(f"| {rep} | {old_score:.4f} | {metric['accuracy']:.4f} | {metric['macro_precision']:.4f} | {metric['macro_recall']:.4f} | {metric['macro_f1']:.4f} |")
    lines += ["", "All comparisons above reuse the same held-out tracks and are descriptive. This test set must not be used to choose among Phase 5B representations or to revise the selected C/gamma values.", "", "## Artifacts and integrity", "", "Phase 5 artifacts were SHA-256 snapshotted before model fitting and verified unchanged after completion. Phase 5B bundles, checkpoint, validation-grid outputs, exploratory-test outputs and report are stored under Phase 5B-specific paths. No audio extraction or other classifier retraining was performed.", "", "## Limitations", "", "This follow-up changes calibration and training-row exposure as well as the decision rule, so a performance difference cannot be attributed solely to the search grid. The test results are exploratory, as stated above. The search is a small prespecified grid and no statistical significance test or repeated-seed analysis was performed.", ""]
    return "\n".join(lines)


def run() -> None:
    config_path = ROOT / "config/config.yaml"
    followup_config_path = ROOT / "config/phase5b_svm.yaml"
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase5b_config = yaml.safe_load(followup_config_path.read_text(encoding="utf-8"))
    if phase5b_config.get("base_config") != "config.yaml":
        raise ValueError("Phase 5B must use the existing base configuration")
    results_dir = ROOT / "results/phase5b_svm"
    metrics_dir = results_dir / "metrics"
    model_dir = ROOT / "models/phase5b_svm"
    report_path = ROOT / "results/reports/phase5b_svm_followup.md"
    if report_path.exists() and not (results_dir / "phase5b_svm_checkpoint.json").exists():
        assert_fresh_report_path(report_path)

    preflight = _load_preflight(base_config, phase5b_config)
    (phase5_state, cache, feature_path, metadata_path, feature_hash,
     train_ids, validation_ids, y_train, y_validation,
     x_train, x_validation, selected_columns, class_order) = preflight
    grid = candidate_grid(phase5b_config["search"]["C"], phase5b_config["search"]["gamma"])
    if len(grid) != 9 or set(phase5b_config["representations"]) != set(REPRESENTATIONS):
        raise ValueError("Unexpected Phase 5B grid/representation specification")
    run_signature = stable_hash({
        "phase5_run_signature": phase5_state["run_signature"],
        "feature_sha256": feature_hash,
        "phase5b_config": phase5b_config,
        "base_config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__)),
        "helper_sha256": _sha256(ROOT / "src/svm_followup.py"),
        "sklearn_version": sklearn.__version__,
    })
    checkpoint_path = results_dir / "phase5b_svm_checkpoint.json"
    state = load_or_create_checkpoint(checkpoint_path, run_signature=run_signature,
                                      results_dir=results_dir, models_dir=model_dir)
    _verify_protected_hashes(results_dir)
    log_path = results_dir / "phase5b_svm.log"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="a", encoding="utf-8")])
    LOG.info("Phase 5B preflight passed: %d train / %d validation rows; %d features; %d candidates per representation",
             len(train_ids), len(validation_ids), len(FEATURE_COLUMNS), len(grid))

    candidate_count_by_rep = {rep: _candidate_records(state, rep) for rep in REPRESENTATIONS}
    for rep in REPRESENTATIONS:
        if rep in state["selected"]:
            artifact = Path(state["selected"][rep]["artifact"])
            _check_bundle(artifact, rep, state["selected"][rep], class_order,
                          feature_hash, len(train_ids), x_train[rep].shape[1])
            continue
        cached = candidate_count_by_rep[rep]
        winning_estimator = None
        winning_index = None
        winning_score = float("-inf")
        for index, params in enumerate(grid):
            if index in cached:
                continue
            estimator, rec = fit_candidate(x_train[rep], y_train,
                                           x_validation[rep], y_validation,
                                           class_order, params)
            candidate = _make_record(estimator, params, rep, index, rec)
            state["candidate_results"].append(candidate)
            cached[index] = candidate
            atomic_json(checkpoint_path, state)
            LOG.info("Validation candidate %s %d/9 C=%s gamma=%s macro-F1=%.5f fit=%.1fs",
                     rep, index + 1, params["C"], params["gamma"],
                     candidate["validation_macro_f1"], candidate["training_seconds"])
            if candidate["validation_macro_f1"] > winning_score:
                winning_estimator = estimator
                winning_index = index
                winning_score = candidate["validation_macro_f1"]
            else:
                del estimator
        if set(cached) != set(range(len(grid))):
            raise RuntimeError(f"Candidate checkpoint incomplete for {rep}")
        best = max((cached[i] for i in range(len(grid))),
                   key=lambda row: row["validation_macro_f1"])
        params = {"C": float(best["C"]), "gamma": best["gamma"]}
        bundle_path = model_dir / f"phase5b_svm__{rep}.joblib"
        if bundle_path.exists():
            bundle = joblib.load(bundle_path)
            if bundle.get("run_signature") != run_signature or bundle.get("selected_hyperparameters") != params:
                raise ValueError(f"Uncheckpointed bundle does not match validation winner: {bundle_path}")
            estimator = bundle["model"]
            refit_seconds = float(bundle["final_refit_seconds"])
        else:
            if winning_estimator is not None and winning_index == int(best["candidate_index"]):
                estimator = winning_estimator
                refit_seconds = float(best["training_seconds"])
            else:
                # A resumed run may have all candidate metrics checkpointed
                # but no in-memory estimator. Refit only the selected candidate
                # in that recovery case; a clean run saves the winning one of
                # the nine already-fitted candidates without an extra fit.
                estimator, final_rec = fit_candidate(x_train[rep], y_train,
                                                     x_validation[rep], y_validation,
                                                     class_order, params)
                refit_seconds = float(final_rec["training_seconds"])
                if not np.isclose(final_rec["validation_metrics"]["macro_f1"],
                                  best["validation_macro_f1"], rtol=0, atol=1e-12):
                    raise RuntimeError(f"Deterministic winner refit score changed for {rep}")
            representation_info = cache["representation_info"][rep]
            transformer = cache["representations"][rep][3]
            bundle = {
                "schema_version": 1, "feature_columns": list(FEATURE_COLUMNS),
                "selected_columns": selected_columns[rep], "labels": class_order,
                "audio_config": base_config["audio"], "representation": rep,
                "transformer": transformer, "model": estimator,
                "selected_hyperparameters": params,
                "final_refit_seconds": refit_seconds,
                "training_rows": len(train_ids), "validation_rows": len(validation_ids),
                "input_dimension": int(x_train[rep].shape[1]),
                "pca_retained_components": representation_info.get("retained_components"),
                "pca_explained_variance": representation_info.get("explained_variance"),
                "calibration": False, "class_weight": None,
                "phase5b_feature_sha256": feature_hash,
                "run_signature": run_signature,
            }
            save_new_bundle(bundle_path, bundle)
        # Recompute final validation metrics from the exact bundle estimator.
        start = time.perf_counter(); val_pred = estimator.predict(x_validation[rep])
        val_inference = time.perf_counter() - start
        val_metrics = evaluate_predictions(y_validation, val_pred, class_order,
                                           val_inference, len(val_pred))
        if not np.isclose(val_metrics["macro_f1"], best["validation_macro_f1"],
                          rtol=0, atol=1e-12):
            raise RuntimeError(f"Saved final SVC validation score differs for {rep}")
        representation_info = cache["representation_info"][rep]
        state["selected"][rep] = {
            "representation": rep, "selected_hyperparameters": params,
            "validation_metrics": val_metrics,
            "validation_macro_f1": val_metrics["macro_f1"],
            "training_rows": len(train_ids), "validation_rows": len(validation_ids),
            "input_dimension": int(x_train[rep].shape[1]),
            "selected_columns": selected_columns[rep],
            "pca_retained_components": representation_info.get("retained_components"),
            "pca_explained_variance": representation_info.get("explained_variance"),
            "final_refit_seconds": refit_seconds,
            "artifact": str(bundle_path),
        }
        atomic_json(checkpoint_path, state)
        LOG.info("Selected %s C=%s gamma=%s validation macro-F1=%.5f",
                 rep, params["C"], params["gamma"], val_metrics["macro_f1"])

    if set(state["selected"]) != set(REPRESENTATIONS):
        raise RuntimeError("Not all four representation SVMs have validation selections")
    _save_validation_outputs(results_dir, state, phase5b_config)
    for rep in REPRESENTATIONS:
        chosen = state["selected"][rep]
        _check_bundle(Path(chosen["artifact"]), rep, chosen, class_order,
                      feature_hash, len(train_ids), x_train[rep].shape[1])
    state["stage"] = "validation_selection_complete"
    state["validation_completed_at_epoch_seconds"] = time.time()
    atomic_json(checkpoint_path, state)

    # No test labels/features are loaded until every selected bundle is saved,
    # checked and represented in the validation-complete checkpoint.
    if state["stage"] not in ("validation_selection_complete", "exploratory_test_evaluation", "complete"):
        raise RuntimeError("Test evaluation gate not reached after validation selection")
    if len(state["selected"]) != 4 or state.get("test_in_progress"):
        raise RuntimeError("Refusing test access before all validation selections are complete")
    test_ids = pd.read_csv(feature_path, usecols=["track_id", "split"])
    expected_test_ids = test_ids.loc[test_ids.split.eq("test"), "track_id"].astype(int).tolist()
    if len(expected_test_ids) != 1772:
        raise ValueError("Unexpected test feature row count")
    if state["stage"] != "complete":
        state["stage"] = "exploratory_test_evaluation"
        atomic_json(checkpoint_path, state)
    if len(state["exploratory_test_results"]) < len(REPRESENTATIONS):
        test = _test_frame_after_selection(feature_path, metadata_path, feature_hash,
                                           set(train_ids) | set(validation_ids),
                                           expected_test_ids,
                                           base_config["project"]["dataset_subset"])
        for rep in REPRESENTATIONS:
            if rep in state["exploratory_test_results"]:
                continue
            state["test_in_progress"] = rep
            atomic_json(checkpoint_path, state)
            bundle = joblib.load(state["selected"][rep]["artifact"])
            values = test.loc[:, bundle["selected_columns"]]
            if bundle["transformer"] is not None:
                values = bundle["transformer"].transform(values)
            start = time.perf_counter(); pred = bundle["model"].predict(values)
            inference_seconds = time.perf_counter() - start
            state["exploratory_test_results"][rep] = evaluate_predictions(
                test.genre, pred, class_order, inference_seconds, len(pred))
            state["test_in_progress"] = None
            atomic_json(checkpoint_path, state)
            LOG.info("Exploratory test metrics saved for %s; macro-F1=%.5f",
                     rep, state["exploratory_test_results"][rep]["macro_f1"])
    if set(state["exploratory_test_results"]) != set(REPRESENTATIONS):
        raise RuntimeError("Exploratory test records incomplete")
    state["stage"] = "complete"
    state["completed_at_epoch_seconds"] = time.time()
    atomic_json(checkpoint_path, state)
    atomic_json(metrics_dir / "exploratory_test_metrics.json", state["exploratory_test_results"])
    _atomic_csv(metrics_dir / "exploratory_test_metrics.csv", [
        {"representation": rep, **{k: v for k, v in state["exploratory_test_results"][rep].items()
                                    if k not in ("confusion_matrix", "per_class")}}
        for rep in REPRESENTATIONS
    ])
    if not report_path.exists():
        report = _report_text(state, phase5_state, base_config,
                              ROOT / "results/metrics/experiment_results.csv", phase5b_config)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temp = report_path.with_suffix(".md.tmp")
        temp.write_text(report, encoding="utf-8")
        if report_path.exists():
            raise FileExistsError(f"Refusing to overwrite Phase 5B report: {report_path}")
        os.replace(temp, report_path)
    _verify_protected_hashes(results_dir)
    atomic_json(results_dir / "audit/phase5_final_sha256.json", _protected_hashes())
    LOG.info("Phase 5B complete; all Phase 5 protected files remain unchanged")


if __name__ == "__main__":
    run()
