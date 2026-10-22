"""Consistent metrics and inference timing for validation and held-out test."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)


def evaluate_predictions(y_true, y_pred, class_order: list[str], elapsed_seconds: float,
                        n_predictions: int, average: str = "macro") -> dict[str, Any]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=class_order,
                                                  average=average, zero_division=0)
    per_class = classification_report(y_true, y_pred, labels=class_order,
                                      output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
        "inference_seconds": float(elapsed_seconds),
        "inference_ms_per_track": float(elapsed_seconds * 1000 / max(1, n_predictions)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_order).tolist(),
        "per_class": {name: per_class[name] for name in class_order if name in per_class},
    }


def predict_timed(model, X) -> tuple[np.ndarray, float]:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    start = time.perf_counter()
    predictions = model.predict(X)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    return np.asarray(predictions), time.perf_counter() - start


def results_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in row.items() if k not in ("confusion_matrix", "per_class")}
                         for row in results])

