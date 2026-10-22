"""FMA metadata loading, ID mapping, split validation, and audit helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [("set", "subset"), ("set", "split"), ("track", "genre_top")]


def track_audio_path(audio_root: str | Path, track_id: int) -> Path:
    """Map FMA numeric ID to its canonical 3-digit directory and 6-digit MP3."""
    track_id = int(track_id)
    if track_id < 0 or track_id > 999_999:
        raise ValueError(f"Invalid FMA track ID: {track_id}")
    text = f"{track_id:06d}"
    return Path(audio_root) / text[:3] / f"{text}.mp3"


def load_track_metadata(tracks_csv: str | Path, audio_root: str | Path,
                        subset: str = "medium", require_audio: bool = False) -> pd.DataFrame:
    """Return labeled subset rows keyed by track_id with official split and audio path."""
    frame = pd.read_csv(tracks_csv, header=[0, 1], index_col=0)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"tracks.csv is missing required columns: {missing}")
    if not frame.index.is_unique:
        duplicates = frame.index[frame.index.duplicated()].unique().tolist()[:10]
        raise ValueError(f"Duplicate track IDs in metadata: {duplicates}")
    rows = frame.loc[frame[("set", "subset")].eq(subset)].copy()
    rows.index.name = "track_id"
    rows["genre"] = rows[("track", "genre_top")]
    rows["split"] = rows[("set", "split")]
    rows["audio_path"] = [str(track_audio_path(audio_root, tid)) for tid in rows.index]
    # Flatten FMA's two-level CSV headers before returning the analysis table;
    # otherwise row access by these metadata keys returns a partial DataFrame.
    if isinstance(rows.columns, pd.MultiIndex):
        rows.columns = [str(first) if second in ("", None) else f"{first}_{second}"
                        for first, second in rows.columns]
    if rows["genre"].isna().any():
        raise ValueError(f"{int(rows['genre'].isna().sum())} {subset} tracks have no top-level genre label")
    allowed = {"training", "validation", "test"}
    bad_splits = sorted(set(rows["split"].dropna()) - allowed)
    if rows["split"].isna().any() or bad_splits:
        raise ValueError(f"Unexpected or missing official split values: {bad_splits}")
    if require_audio:
        exists = rows["audio_path"].map(lambda p: Path(p).is_file())
        rows = rows.loc[exists].copy()
    return rows


def official_partitions(metadata: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split rows strictly by official FMA split labels and assert disjoint IDs."""
    parts = {name: metadata.loc[metadata["split"].eq(name)].copy()
             for name in ("training", "validation", "test")}
    id_sets = [set(part.index) for part in parts.values()]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Official partitions overlap by track ID")
    if sum(map(len, parts.values())) != len(metadata):
        raise ValueError("Official partitions do not account for every metadata row")
    return parts


def audit_dataset(metadata: pd.DataFrame) -> dict[str, Any]:
    """Summarize rows and report absent audio without silently dropping records."""
    parts = official_partitions(metadata)
    missing_audio = metadata.loc[~metadata["audio_path"].map(lambda p: Path(p).is_file())]
    return {
        "metadata_tracks": int(len(metadata)),
        "missing_audio_tracks": int(len(missing_audio)),
        "missing_audio_ids": [int(x) for x in missing_audio.index],
        "split_counts": {name: int(len(part)) for name, part in parts.items()},
        "genre_counts": {str(k): int(v) for k, v in metadata["genre"].value_counts().sort_index().items()},
        "split_genre_counts": {name: {str(k): int(v) for k, v in part["genre"].value_counts().sort_index().items()}
                               for name, part in parts.items()},
    }

