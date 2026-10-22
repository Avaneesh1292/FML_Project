"""Tiny, in-memory train/validation smoke test; never reads the test split."""
from __future__ import annotations
from pathlib import Path
import json
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loading import load_track_metadata
from src.dimensionality_reduction import PCATransformer
from src.evaluation import evaluate_predictions
from src.feature_extraction import FEATURE_COLUMNS, extract_feature_vector
from src.feature_selection import mutual_information_ranking, random_forest_ranking, select_top_features
from src.model_training import train_mlp, train_sklearn
from src.utils import configure_logging, load_config, project_path, set_seed


def main():
    configure_logging()
    config = load_config()
    set_seed(int(config["project"]["seed"]))
    # Four well represented genres; three train examples/class and one validation example/class.
    genres = ["Classical", "Electronic", "Hip-Hop", "Rock"]
    train_per_class, validation_per_class = 3, 1
    metadata = load_track_metadata(project_path(config, config["paths"]["tracks_csv"]),
                                  project_path(config, config["paths"]["audio_dir"]), subset="medium")
    selected = []
    for genre in genres:
        group = metadata.loc[metadata["genre"].eq(genre)]
        training = group.loc[group["split"].eq("training")].sort_index().head(train_per_class)
        validation = group.loc[group["split"].eq("validation")].sort_index().head(validation_per_class)
        if len(training) != train_per_class or len(validation) != validation_per_class:
            raise ValueError(f"Insufficient official train/validation tracks for {genre}")
        selected.extend(("training", row) for _, row in training.iterrows())
        selected.extend(("validation", row) for _, row in validation.iterrows())

    records = []
    extraction_start = time.perf_counter()
    for index, (partition, row) in enumerate(selected, start=1):
        features = extract_feature_vector(row["audio_path"], config["audio"])
        if len(features) != 74 or not all(pd.notna(value) and abs(value) != float("inf")
                                          for value in features.values()):
            raise ValueError(f"Invalid feature vector for track {row.name}")
        records.append({"track_id": int(row.name), "genre": row["genre"],
                        "split": partition, **features})
        print(f"extracted {index}/{len(selected)} ID={row.name} split={partition} genre={row['genre']}", flush=True)
    extraction_seconds = time.perf_counter() - extraction_start
    frame = pd.DataFrame(records)
    train = frame.loc[frame["split"].eq("training")]
    validation = frame.loc[frame["split"].eq("validation")]
    X_train, X_validation = train[FEATURE_COLUMNS], validation[FEATURE_COLUMNS]
    y_train, y_validation = train["genre"], validation["genre"]

    mi_names = select_top_features(mutual_information_ranking(X_train, y_train, int(config["project"]["seed"])), 10)
    rf_ranking = random_forest_ranking(X_train, y_train, int(config["project"]["seed"]), n_estimators=20)
    rf_names = select_top_features(rf_ranking, 10)
    pca = PCATransformer(0.95).fit(X_train)
    representations = {
        "full": (X_train, X_validation, None),
        "mutual_information": (X_train[mi_names], X_validation[mi_names], None),
        "random_forest": (X_train[rf_names], X_validation[rf_names], None),
        "pca_95": (pca.transform(X_train), pca.transform(X_validation), pca),
    }
    config["models"]["random_forest"]["n_estimators"] = 20
    config["models"]["mlp"].update({"hidden_layers": [32], "dropout": 0.1,
                                     "max_epochs": 3, "patience": 2, "batch_size": 4})
    results = []
    for representation, (xtr, xva, transformer) in representations.items():
        for model_name in ("logistic_regression", "svm", "random_forest", "mlp"):
            if model_name == "mlp":
                fitted = train_mlp(xtr, y_train, xva, y_validation, config, int(config["project"]["seed"]))
                training_seconds = fitted.training_seconds
            else:
                fitted, training_seconds = train_sklearn(model_name, xtr, y_train, config,
                                                         int(config["project"]["seed"]))
            prediction = fitted.predict(xva)
            metrics = evaluate_predictions(y_validation, prediction, genres, 0.0, len(prediction))
            results.append({"representation": representation, "model": model_name,
                            "device": getattr(fitted, "device", "scikit-learn/CPU"),
                            "fit_seconds": training_seconds,
                            "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
    print(json.dumps({"scope": "smoke only; official train and validation; test split not loaded",
                      "tracks": len(frame), "train_tracks": len(train),
                      "validation_tracks": len(validation), "features_per_track": len(FEATURE_COLUMNS),
                      "extraction_seconds": extraction_seconds,
                      "pca_components": int(pca.pca.n_components_),
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()

