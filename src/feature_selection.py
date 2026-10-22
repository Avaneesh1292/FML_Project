"""Train-only feature ranking and selection utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def _valid_k(k: int, n_features: int) -> int:
    if n_features < 1:
        raise ValueError("At least one input feature is required")
    if not 1 <= int(k) <= n_features:
        raise ValueError(f"k must be in [1, {n_features}], got {k}")
    return int(k)


def mutual_information_ranking(X_train: pd.DataFrame, y_train: pd.Series,
                               seed: int = 42) -> pd.Series:
    scores = mutual_info_classif(X_train, y_train, random_state=seed)
    return pd.Series(scores, index=X_train.columns, name="mutual_information").sort_values(ascending=False)


def select_mutual_information(X_train: pd.DataFrame, y_train: pd.Series, k: int,
                              seed: int = 42) -> list[str]:
    k = _valid_k(k, X_train.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.set_params(score_func=lambda x, y: mutual_info_classif(x, y, random_state=seed))
    selector.fit(X_train, y_train)
    return list(X_train.columns[selector.get_support()])


def random_forest_ranking(X_train: pd.DataFrame, y_train: pd.Series,
                          seed: int = 42, n_estimators: int = 300) -> pd.Series:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed,
                                   n_jobs=-1, class_weight="balanced_subsample")
    model.fit(X_train, y_train)
    return pd.Series(model.feature_importances_, index=X_train.columns,
                     name="random_forest_importance").sort_values(ascending=False)


def select_top_features(ranking: pd.Series, k: int) -> list[str]:
    _valid_k(k, len(ranking))
    return list(ranking.iloc[:k].index)


def candidate_counts(n_features: int, fractions: list[float] | None = None) -> list[int]:
    fractions = fractions or [0.25, 0.50, 0.75]
    counts = {max(1, min(n_features, int(np.ceil(n_features * f)))) for f in fractions}
    return sorted(counts)

