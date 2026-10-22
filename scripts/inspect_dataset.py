"""Read-only summary of FMA Medium metadata, splits, files, and class balance."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loading import audit_dataset, load_track_metadata
from src.plotting import plot_class_distribution
from src.utils import configure_logging, load_config, project_path


def main():
    configure_logging()
    config = load_config()
    metadata = load_track_metadata(project_path(config, config["paths"]["tracks_csv"]),
                                  project_path(config, config["paths"]["audio_dir"]),
                                  subset=config["project"]["dataset_subset"])
    print(json.dumps(audit_dataset(metadata), indent=2))
    plot_class_distribution(metadata, project_path(config, config["paths"]["results_dir"]) /
                            "figures" / "class_distribution.png")


if __name__ == "__main__":
    main()

