# Music genre classifier web app

Streamlit entry point: `website/app.py` (repository root is the working directory).

The SVM uses the existing `models/phase5b_svm/phase5b_svm__mutual_information.joblib` bundle. The CPU MLP checkpoint is `website/models/phase5_mutual_information_mlp_cpu.pt`; it is an inference-only export of the saved Phase 5 Mutual Information + MLP bundle and contains the original 37-feature order, training-fitted scaler statistics, labels, audio configuration, and CPU state dictionary.

To regenerate that export on the project workstation, use the configured `fml-music` environment:

```powershell
& "$env:USERPROFILE\anaconda3\envs\fml-music\python.exe" website/export_mlp_checkpoint.py
```

The exporter refuses to overwrite an existing export. It does not train or modify the source model. To launch locally:

```powershell
& "$env:USERPROFILE\anaconda3\envs\fml-music\python.exe" -m streamlit run website/app.py
```

For Streamlit Community Cloud, deploy the repository with `website/app.py` as the app entry point and include `website/requirements.txt`. Inference uses CPU; the MLP imports PyTorch only when that model is selected. The app requires at least 30 seconds of decodable audio and predicts one of the 16 FMA Medium labels. SVM confidence probabilities are unavailable.
