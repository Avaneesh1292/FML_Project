"""Audio utilities for the Streamlit web application.

Bridges uploaded audio files with the existing prediction pipeline
by writing selected segments to temporary WAV files. The existing
predict_audio() function is called unchanged.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def load_audio_from_upload(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """Decode uploaded audio bytes into a mono waveform at 22 050 Hz.

    Uses librosa for decoding — the same library as the training pipeline.

    Returns
    -------
    dict with keys: waveform, sr, duration, channels, original_sr
    """
    suffix = Path(filename).suffix.lower()
    if not suffix:
        raise ValueError("Uploaded file has no extension; cannot determine audio format.")
    allowed = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma"}
    if suffix not in allowed:
        raise ValueError(
            f"Unsupported audio format '{suffix}'. "
            f"Accepted formats: {', '.join(sorted(allowed))}"
        )

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(file_bytes)

        if os.path.getsize(tmp_path) == 0:
            raise ValueError("Uploaded file is empty (0 bytes).")

        # Retrieve original channel / sample-rate info before librosa resamples
        try:
            info = sf.info(tmp_path)
            channels = info.channels
            original_sr = info.samplerate
        except Exception:
            channels = 1
            original_sr = 0

        # Decode to mono @ 22 050 Hz — matching the training pipeline exactly
        import librosa

        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
        y = np.asarray(y, dtype=np.float32)

        if y.ndim != 1 or y.size == 0:
            raise ValueError("Audio decoding produced an empty or invalid waveform.")
        if not np.isfinite(y).all():
            raise ValueError("Decoded audio contains invalid (NaN / infinite) values.")

        duration = float(len(y) / sr)

        return {
            "waveform": y,
            "sr": int(sr),
            "duration": duration,
            "channels": int(channels),
            "original_sr": int(original_sr),
        }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not decode audio file: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def extract_segment(
    waveform: np.ndarray,
    sr: int,
    start_sec: float,
    duration_sec: float = 30.0,
) -> np.ndarray:
    """Extract exactly *duration_sec* seconds starting at *start_sec*.

    Raises ValueError if the requested window falls outside the waveform.
    """
    total_duration = len(waveform) / sr
    if start_sec < 0:
        raise ValueError("Start time cannot be negative.")
    if start_sec + duration_sec > total_duration + 0.05:  # 50 ms tolerance
        raise ValueError(
            f"Segment [{start_sec:.1f}s – {start_sec + duration_sec:.1f}s] "
            f"exceeds audio duration ({total_duration:.1f}s)."
        )

    start_sample = int(round(start_sec * sr))
    end_sample = start_sample + int(round(duration_sec * sr))
    end_sample = min(end_sample, len(waveform))

    segment = waveform[start_sample:end_sample]
    if len(segment) == 0:
        raise ValueError("Extracted segment is empty.")

    return segment


def segment_to_wav_bytes(segment: np.ndarray, sr: int) -> bytes:
    """Convert a waveform segment to WAV bytes for ``st.audio()`` playback."""
    buffer = io.BytesIO()
    sf.write(buffer, segment, sr, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


def predict_segment_genre(
    segment: np.ndarray, sr: int, bundle: dict[str, Any]
) -> dict[str, Any]:
    """Write *segment* to a temporary WAV, call the existing pipeline, clean up.

    This is the **only** bridge between in-memory segments and the
    file-path-based ``predict_audio()`` function.  The existing ML pipeline
    (feature extraction, feature selection, scaling, SVM prediction) runs
    completely unchanged.
    """
    from src.prediction import predict_audio  # lazy — sys.path set in app.py

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        os.close(fd)  # close the raw fd so soundfile can open the path
        sf.write(tmp_path, segment, sr, format="WAV", subtype="PCM_16")
        result = predict_audio(tmp_path, bundle)
        return result
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def generate_waveform_data(
    waveform: np.ndarray, sr: int, target_points: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """Down-sample *waveform* for visualisation.

    Creates a min/max envelope so peaks are preserved even at low resolution.

    Returns
    -------
    (times, amplitudes) – both 1-D float arrays
    """
    total_samples = len(waveform)

    if total_samples <= target_points * 2:
        times = np.arange(total_samples, dtype=np.float64) / sr
        return times, waveform.astype(np.float64)

    bin_size = total_samples // target_points
    n_bins = total_samples // bin_size
    trimmed = waveform[: n_bins * bin_size].reshape(n_bins, bin_size)

    maxs = trimmed.max(axis=1)
    mins = trimmed.min(axis=1)

    # Interleave max → min for an envelope shape
    envelope = np.empty(n_bins * 2, dtype=np.float32)
    envelope[0::2] = maxs
    envelope[1::2] = mins

    times = np.linspace(0, n_bins * bin_size / sr, n_bins * 2, dtype=np.float64)
    return times, envelope.astype(np.float64)


# ── formatting helpers ──────────────────────────────────────────────────────


def format_duration(seconds: float) -> str:
    """Format seconds as ``MM:SS``."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def format_file_size(size_bytes: int) -> str:
    """Format byte count in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"
