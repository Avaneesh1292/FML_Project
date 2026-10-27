"""Export the existing Phase 5 MLP weights as a CPU-safe inference checkpoint.

Run from the project root with the training environment's Python. The source
joblib bundle is read only; this script never retrains or writes to it.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys

import joblib
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "models" / "mutual_information__mlp.joblib"
OUTPUT = Path(__file__).resolve().parent / "models" / "phase5_mutual_information_mlp_cpu.pt"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing MLP export: {OUTPUT}")
    source_bytes = SOURCE.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    bundle = joblib.load(io.BytesIO(source_bytes))
    if bundle.get("representation") != "mutual_information" or len(bundle.get("selected_columns", [])) != 37:
        raise ValueError("Source bundle is not the expected Phase 5 MI MLP bundle")
    result = bundle["model"]
    if list(result.feature_names) != list(bundle["selected_columns"]):
        raise ValueError("MLP fitted feature order differs from saved selected_columns")
    if list(map(str, result.classes_)) != list(bundle["labels"]):
        raise ValueError("MLP fitted class order differs from saved labels")
    if result.scaler.n_features_in_ != 37:
        raise ValueError("MLP scaler input dimension is not 37")
    model = result.model
    state = {name: tensor.detach().to(device="cpu").clone()
             for name, tensor in model.state_dict().items()}
    architecture = {
        "input_dim": len(bundle["selected_columns"]),
        "output_dim": len(bundle["labels"]),
        "hidden_layers": [256, 128],
        "dropout": 0.3,
        "activation": "relu",
    }
    scaler = result.scaler
    payload = {
        "format_version": 1,
        "source_sha256": digest,
        "feature_columns": list(bundle["feature_columns"]),
        "selected_columns": list(bundle["selected_columns"]),
        "labels": list(bundle["labels"]),
        "audio_config": dict(bundle["audio_config"]),
        "architecture": architecture,
        "state_dict": state,
        "scaler": {
            "mean": torch.as_tensor(scaler.mean_.copy(), dtype=torch.float64),
            "scale": torch.as_tensor(scaler.scale_.copy(), dtype=torch.float64),
            "var": torch.as_tensor(scaler.var_.copy(), dtype=torch.float64),
            "n_features_in": int(scaler.n_features_in_),
            "n_samples_seen": int(scaler.n_samples_seen_),
        },
        "best_epoch": int(result.best_epoch),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUTPUT)
    print(f"Exported CPU MLP checkpoint: {OUTPUT}")
    print(f"Source SHA-256: {digest}")


if __name__ == "__main__":
    main()
