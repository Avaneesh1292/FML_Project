"""Use and validate the official FMA splits; no random reassignment by default."""
from __future__ import annotations

import pandas as pd

from .data_loading import official_partitions


def split_feature_table(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {"track_id", "genre", "split"}
    if not required.issubset(features.columns):
        raise ValueError(f"Feature table missing metadata columns: {sorted(required - set(features.columns))}")
    if features["track_id"].duplicated().any():
        raise ValueError("Feature table contains duplicate track IDs")
    indexed = features.set_index("track_id", drop=True)
    return official_partitions(indexed)

