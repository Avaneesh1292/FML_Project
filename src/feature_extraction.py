"""Fixed-schema summary acoustic features and restartable CSV cache extraction."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .audio_preprocessing import load_audio
from .utils import atomic_json, stable_hash

LOG = logging.getLogger(__name__)
FEATURE_COLUMNS = (
    [f"mfcc_{i:02d}_{stat}" for i in range(1, 21) for stat in ("mean", "std")]
    + [f"chroma_{i:02d}_{stat}" for i in range(1, 13) for stat in ("mean", "std")]
    + [f"{name}_{stat}" for name in ("zcr", "spectral_centroid", "spectral_bandwidth",
                                        "spectral_rolloff", "rms") for stat in ("mean", "std")]
)
IDENTIFIER_COLUMNS = ["track_id", "genre", "split"]


def feature_config(config: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": config["features"]["schema_version"], **config["audio"]}


def extract_feature_vector(path: str | Path, audio_cfg: dict[str, Any]) -> dict[str, float]:
    y, sr = load_audio(path, sample_rate=int(audio_cfg["sample_rate"]), mono=bool(audio_cfg["mono"]))
    import librosa
    n_fft, hop = int(audio_cfg["n_fft"]), int(audio_cfg["hop_length"])
    if y.size < n_fft:
        raise ValueError(f"Audio shorter than analysis window ({y.size} samples < {n_fft})")
    kwargs = {"n_fft": n_fft, "hop_length": hop}
    arrays: list[np.ndarray] = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=int(audio_cfg["n_mfcc"]), **kwargs)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=int(audio_cfg["n_chroma"]), **kwargs)
    arrays.extend([mfcc, chroma])
    arrays.extend([
        librosa.feature.zero_crossing_rate(y, hop_length=hop),
        librosa.feature.spectral_centroid(y=y, sr=sr, **kwargs),
        librosa.feature.spectral_bandwidth(y=y, sr=sr, **kwargs),
        librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=float(audio_cfg["roll_percent"]), **kwargs),
        librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop),
    ])
    values: dict[str, float] = {}
    families = [("mfcc", 20), ("chroma", 12)]
    for prefix, family, count in (("mfcc", arrays[0], 20), ("chroma", arrays[1], 12)):
        for i in range(count):
            values[f"{prefix}_{i+1:02d}_mean"] = float(np.mean(family[i]))
            values[f"{prefix}_{i+1:02d}_std"] = float(np.std(family[i]))
    for name, family in zip(("zcr", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "rms"), arrays[2:]):
        values[f"{name}_mean"] = float(np.mean(family))
        values[f"{name}_std"] = float(np.std(family))
    if list(values) != FEATURE_COLUMNS:
        raise RuntimeError("Internal feature schema ordering mismatch")
    vector = np.asarray(list(values.values()), dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("Extracted features contain NaN or infinite values")
    return values


def extract_dataset(metadata: pd.DataFrame, audio_cfg: dict[str, Any], feature_cfg: dict[str, Any],
                    output_csv: str | Path, failures_csv: str | Path, manifest_path: str | Path,
                    checkpoint_every: int = 100, limit: int | None = None) -> pd.DataFrame:
    """Extract one track at a time; write periodic snapshots and resume matching cache."""
    output_csv, failures_csv, manifest_path = map(Path, (output_csv, failures_csv, manifest_path))
    signature = stable_hash({"feature_config": feature_cfg, "columns": FEATURE_COLUMNS})
    if manifest_path.exists():
        import json
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise ValueError("Existing feature cache uses a different configuration/schema; move it aside explicitly")
    elif output_csv.exists():
        raise ValueError("Feature cache exists without a schema manifest; move it aside explicitly before reuse")
    atomic_json(manifest_path, {"signature": signature, "feature_config": feature_cfg,
                                "feature_columns": FEATURE_COLUMNS, "status": "in_progress"})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    successes: dict[int, dict[str, Any]] = {}
    failures: dict[int, dict[str, Any]] = {}
    if output_csv.exists():
        cached = pd.read_csv(output_csv)
        if not set(IDENTIFIER_COLUMNS + FEATURE_COLUMNS).issubset(cached.columns):
            raise ValueError("Feature cache columns do not match expected schema")
        successes = {int(r.track_id): r._asdict() for r in cached.itertuples(index=False)}
    if failures_csv.exists():
        old = pd.read_csv(failures_csv)
        failures = {int(r.track_id): {"track_id": int(r.track_id), "reason": str(r.reason)}
                    for r in old.itertuples(index=False)}
    rows = metadata if limit is None else metadata.head(int(limit))
    changed = 0
    pending_successes: list[dict[str, Any]] = []
    pending_failures: list[dict[str, Any]] = []
    for track_id, row in tqdm(rows.iterrows(), total=len(rows), desc="Extracting audio features"):
        track_id = int(track_id)
        if track_id in successes or track_id in failures:
            continue
        try:
            features = extract_feature_vector(row["audio_path"], audio_cfg)
            record = {"track_id": track_id, "genre": row["genre"], "split": row["split"], **features}
            successes[track_id] = record
            pending_successes.append(record)
            failures.pop(track_id, None)
        except Exception as exc:
            LOG.exception("Feature extraction failed for track %s", track_id)
            failures[track_id] = {"track_id": track_id, "reason": f"{type(exc).__name__}: {exc}"}
            pending_failures.append(failures[track_id])
        changed += 1
        if changed % max(1, checkpoint_every) == 0:
            _append_cache(pending_successes, pending_failures, output_csv, failures_csv)
            pending_successes.clear()
            pending_failures.clear()
    _append_cache(pending_successes, pending_failures, output_csv, failures_csv)
    if not output_csv.exists():
        pd.DataFrame(columns=IDENTIFIER_COLUMNS + FEATURE_COLUMNS).to_csv(output_csv, index=False)
    if not failures_csv.exists():
        pd.DataFrame(columns=["track_id", "reason"]).to_csv(failures_csv, index=False)
    atomic_json(manifest_path, {"signature": signature, "feature_config": feature_cfg,
                                "feature_columns": FEATURE_COLUMNS, "successful_tracks": len(successes),
                                "failed_tracks": len(failures)})
    return pd.DataFrame(successes.values(), columns=IDENTIFIER_COLUMNS + FEATURE_COLUMNS)


def _append_cache(successes: list[dict[str, Any]], failures: list[dict[str, Any]],
                  output_csv: Path, failures_csv: Path) -> None:
    if successes:
        pd.DataFrame(successes, columns=IDENTIFIER_COLUMNS + FEATURE_COLUMNS).to_csv(
            output_csv, mode="a", header=not output_csv.exists(), index=False)
    if failures:
        pd.DataFrame(failures, columns=["track_id", "reason"]).to_csv(
            failures_csv, mode="a", header=not failures_csv.exists(), index=False)

