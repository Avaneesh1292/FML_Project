"""Incremental feature extraction; intended for approved runs only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loading import load_track_metadata
from src.feature_extraction import extract_dataset, feature_config
from src.utils import configure_logging, load_config, project_path


def main():
    configure_logging()
    config = load_config()
    metadata = load_track_metadata(project_path(config, config["paths"]["tracks_csv"]),
                                  project_path(config, config["paths"]["audio_dir"]),
                                  subset=config["project"]["dataset_subset"])
    paths = {key: project_path(config, config["paths"][key]) for key in
             ("features_csv", "failures_csv", "feature_manifest")}
    extract_dataset(metadata, config["audio"], feature_config(config), paths["features_csv"],
                    paths["failures_csv"], paths["feature_manifest"],
                    config["features"]["checkpoint_every"], config["features"]["limit"])


if __name__ == "__main__":
    main()

