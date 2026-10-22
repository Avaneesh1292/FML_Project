"""Predict genre for one audio path using a saved fitted bundle."""
from pathlib import Path
import argparse
import json
import sys
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prediction import load_predictor, predict_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = predict_audio(args.audio, load_predictor(args.model))
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

