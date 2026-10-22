"""Safe, deterministic audio loading and basic validation."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

# Numba-backed librosa functions need writable JIT cache storage. Keeping it
# beside generated project artifacts avoids writes into the read-only package.
_NUMBA_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "numba_cache"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE_DIR))


class AudioProcessingError(RuntimeError):
    """Audio could not be loaded or did not meet configured constraints."""


def load_audio(path: str | Path, sample_rate: int = 22050, mono: bool = True,
               min_duration_seconds: float = 0.10, max_duration_seconds: float | None = None
               ) -> tuple[np.ndarray, int]:
    path = Path(path)
    if not path.is_file():
        raise AudioProcessingError(f"Audio file not found: {path}")
    if path.stat().st_size == 0:
        raise AudioProcessingError(f"Audio file is empty: {path}")
    try:
        Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        import librosa
        y, sr = librosa.load(path, sr=sample_rate, mono=mono)
    except Exception as exc:
        raise AudioProcessingError(f"Unable to decode {path}: {exc}") from exc
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or y.size == 0:
        raise AudioProcessingError(f"Expected non-empty mono waveform, got shape {y.shape}")
    if not np.isfinite(y).all():
        raise AudioProcessingError("Waveform contains NaN or infinite values")
    duration = y.size / sr
    if duration < min_duration_seconds:
        raise AudioProcessingError(f"Track too short ({duration:.3f}s; minimum {min_duration_seconds:.3f}s)")
    if max_duration_seconds is not None and duration > max_duration_seconds:
        y = y[: int(max_duration_seconds * sr)]
    return y, int(sr)

