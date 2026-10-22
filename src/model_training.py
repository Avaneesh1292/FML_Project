"""Classical estimators and a CUDA-aware PyTorch tabular MLP."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_sklearn_model(name: str, config: dict[str, Any], seed: int = 42):
    cfg = config["models"]
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=cfg[name]["C"],
                            max_iter=cfg[name]["max_iter"], random_state=seed))
    if name == "svm":
        calibrated = CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", C=cfg[name]["C"], gamma=cfg[name]["gamma"],
                          random_state=seed), method="sigmoid", cv=3)
        return make_pipeline(StandardScaler(), calibrated)
    if name == "random_forest":
        return RandomForestClassifier(**cfg[name], random_state=seed, n_jobs=-1,
                                      class_weight="balanced_subsample")
    raise ValueError(f"Unknown scikit-learn model: {name}")


def train_sklearn(name: str, X_train, y_train, config: dict[str, Any], seed: int = 42):
    estimator = build_sklearn_model(name, config, seed)
    start = time.perf_counter()
    estimator.fit(X_train, y_train)
    return estimator, time.perf_counter() - start


class TabularMLP(nn.Module):
    """Pickleable feed-forward network definition used by training and inference."""
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list[int], dropout: float,
                 activation: str = "relu"):
        super().__init__()
        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        if activation not in acts:
            raise ValueError(f"Unsupported MLP activation {activation!r}")
        layers: list[nn.Module] = []
        dim = input_dim
        for width in hidden_layers:
            layers.extend([nn.Linear(dim, int(width)), acts[activation](), nn.Dropout(dropout)])
            dim = int(width)
        layers.append(nn.Linear(dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


@dataclass
class MLPTrainingResult:
    model: Any
    scaler: Any
    classes: np.ndarray
    device: str
    best_epoch: int
    best_validation_loss: float
    training_seconds: float
    feature_names: list[str]

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.as_tensor(values, dtype=torch.float32, device=self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.classes[self.predict_proba(X).argmax(axis=1)]

    @property
    def classes_(self) -> np.ndarray:
        return self.classes


def train_mlp(X_train: pd.DataFrame, y_train, X_validation: pd.DataFrame, y_validation,
              config: dict[str, Any], seed: int = 42, device: str | None = None) -> MLPTrainingResult:
    """Fit scaler on training only, select epoch by validation loss, and restore best weights."""
    Model = TabularMLP
    cfg = config["models"]["mlp"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    scaler = StandardScaler()
    xtr = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    xva = scaler.transform(X_validation).astype(np.float32, copy=False)
    classes = np.asarray(sorted(pd.unique(y_train)))
    class_to_idx = {label: i for i, label in enumerate(classes)}
    unknown_validation = set(pd.unique(y_validation)) - set(classes)
    if unknown_validation:
        raise ValueError(f"Validation labels absent from training partition: {sorted(unknown_validation)}")
    ytr = np.asarray([class_to_idx[x] for x in y_train], dtype=np.int64)
    yva = np.asarray([class_to_idx[x] for x in y_validation], dtype=np.int64)
    torch_device = torch.device(selected_device)
    model = Model(xtr.shape[1], len(classes), list(cfg["hidden_layers"]), float(cfg["dropout"]),
                  str(cfg.get("activation", "relu"))).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]),
                                 weight_decay=float(cfg["weight_decay"]))
    criterion = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    loader = torch.utils.data.DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True,
                                         generator=generator, num_workers=0)
    val_x = torch.as_tensor(xva, dtype=torch.float32, device=torch_device)
    val_y = torch.as_tensor(yva, dtype=torch.long, device=torch_device)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    start = time.perf_counter()
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(torch_device), yb.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x), val_y).item())
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    if best_state is None:
        raise RuntimeError("MLP training did not produce a model checkpoint")
    model.load_state_dict(best_state)
    return MLPTrainingResult(model, scaler, classes, str(torch_device), best_epoch,
                             best_loss, time.perf_counter() - start,
                             list(X_train.columns) if hasattr(X_train, "columns") else
                             [f"input_{i}" for i in range(X_train.shape[1])])

