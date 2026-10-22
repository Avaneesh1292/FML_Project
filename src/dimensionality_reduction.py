"""Leakage-safe standardization and PCA fitted on training data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCATransformer:
    def __init__(self, variance: float = 0.95):
        if not 0 < variance <= 1:
            raise ValueError("PCA variance threshold must be in (0, 1]")
        self.variance = variance
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance, svd_solver="full")
        self.feature_names_in_: list[str] | None = None

    def fit(self, X_train: pd.DataFrame) -> "PCATransformer":
        self.feature_names_in_ = list(X_train.columns)
        scaled = self.scaler.fit_transform(X_train)
        self.pca.fit(scaled)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_names_in_ is None:
            raise RuntimeError("PCA transformer must be fit on training data first")
        if list(X.columns) != self.feature_names_in_:
            raise ValueError("PCA input columns/order differ from fitted training schema")
        return self.pca.transform(self.scaler.transform(X))

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    @property
    def cumulative_explained_variance_(self) -> np.ndarray:
        return np.cumsum(self.pca.explained_variance_ratio_)

