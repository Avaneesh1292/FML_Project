# FMA Medium Music Genre Classification

This repository implements a controlled feature-based comparison of Logistic Regression, RBF SVM, Random Forest, and a PyTorch MLP using FMA Medium. The project preserves the synopsis objectives and common audio-feature workflow, changes the source dataset from GTZAN to FMA Medium, and adds training-only Mutual Information feature selection, Random Forest feature ranking, and PCA.

## Verified dataset scope

`fma_metadata/tracks.csv` marks 17,000 IDs as `medium`; these map to 17,000 audio files. The audio folder also contains 8,000 `small` tracks. The pipeline filters using the metadata `subset == medium` field and never adds those small tracks. The official split counts are 13,522 training, 1,705 validation, and 1,773 test tracks. Split labels remain in every feature row. Missing audio is reported by the audit and excluded with a recorded reason during extraction.

## Environment

Use the existing Conda environment, not another interpreter:

```powershell
conda activate fml-music
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If activation is unavailable, call the verified interpreter directly:

```powershell
$Python = 'C:\Users\Nalla Avaneesh Reddy\anaconda3\envs\fml-music\python.exe'
& $Python -m unittest discover -s tests -v
```

Dependencies already present in the environment are listed in `requirements-lock.txt`. No Parquet engine is installed, so the restartable feature cache uses CSV. No third-party test runner is required.

## Commands

From the repository root:

```powershell
python scripts/inspect_dataset.py
python scripts/extract_features.py
python scripts/run_experiments.py
python scripts/predict_genre.py path\to\song.mp3 --model models\phase5b_svm\phase5b_svm__mutual_information.joblib
```

Extraction and experiments are intentionally separate. Extraction preserves the complete audio duration, converts to mono, resamples to 22,050 Hz, and writes per-track checkpoints and failures. Numba-backed librosa JIT cache files default to `data/processed/numba_cache` so the runtime does not need to write into the installed package directory. Before an approved full run, inspect durations and estimate runtime/disk use from a representative subset. Feature cache configuration/schema hashes prevent silent reuse after feature changes.

The experiment script keeps the official partitions fixed. Model/transformer fitting uses training rows; MI/RF candidate counts are chosen by validation macro-F1; PCA standardization and PCA are fit on training only. The test partition is evaluated after model selection and is not used for ranking or tuning. The MLP uses mini-batches, validation-loss early stopping, dynamically selects CUDA when present, and supports CPU fallback.

The MLP is a tabular neural network trained with Adam and cross-entropy, mini-batches, and validation early stopping. Logistic Regression, SVM, and Random Forest use scikit-learn estimators; SVM and Logistic Regression include train-fitted standardization. This uses different optimization procedures by model family while keeping data partitions and feature inputs controlled.

## Feature schema

Each track yields 74 predictors: mean and standard deviation of 20 MFCC coefficients, 12 chroma bins, and zero-crossing rate, spectral centroid, spectral bandwidth, spectral roll-off, and RMS. IDs, genre, and official split remain metadata columns, never predictors. Audio decode errors, too-short tracks, and non-finite values are surfaced and recorded.

## Outputs

Generated artifacts are git-ignored. Feature CSVs and failures go under `data/features/`; model bundles under `models/`; result metrics under `results/metrics/`. The full Phase 5 experiments are complete; see `results/reports/phase6_results_analysis.md` for the controlled comparison and `results/reports/phase5b_svm_followup.md` for the SVM follow-up. Phase 5B test results are exploratory because the Phase 5 test results had already been inspected; they are not unbiased final performance estimates.

