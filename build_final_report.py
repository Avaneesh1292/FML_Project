"""Build and verify the FMA Medium academic report PDF from the reviewed draft."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import pdfplumber
from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parent
DRAFT = ROOT / "results/reports/final_report_draft.md"
OUT = ROOT / "results/reports/final_report.pdf"
ASSETS = ROOT / "tmp/final_report_assets"
TITLE = "Automatic Music Genre Classification Using Acoustic Feature Representation and Comparative Model Evaluation"
SUBTITLE = "An Experimental Study Using the FMA Medium Dataset"
SHORT_TITLE = "Automatic Music Genre Classification | FMA Medium"
REP_ORDER = ["full", "mutual_information", "random_forest", "pca_95"]
REP_LABEL = {"full": "Full Features", "mutual_information": "Mutual Information",
             "random_forest": "Random Forest Selection", "pca_95": "PCA (95% Variance)"}
MODEL_ORDER = ["logistic_regression", "svm", "random_forest", "mlp"]
MODEL_LABEL = {"logistic_regression": "Logistic Regression", "svm": "Calibrated SVM",
               "random_forest": "Random Forest", "mlp": "PyTorch MLP"}
GENRES = ["Blues", "Classical", "Country", "Easy Listening", "Electronic", "Experimental",
          "Folk", "Hip-Hop", "Instrumental", "International", "Jazz", "Old-Time / Historic",
          "Pop", "Rock", "Soul-RnB", "Spoken"]

REFERENCES = [
    '[1] G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," *IEEE Transactions on Speech and Audio Processing*, vol. 10, no. 5, pp. 293–302, Jul. 2002, doi: 10.1109/TSA.2002.800560.',
    '[2] M. Ashraf, F. Abid, I. U. Din, J. Rasheed, M. Yesiltepe, S. F. Yeo, and M. T. Ersoy, "A hybrid CNN and RNN variant model for music classification," *Applied Sciences*, vol. 13, no. 3, Art. no. 1476, 2023, doi: 10.3390/app13031476.',
    '[3] Z. Liu, T. Bian, and M. Yang, "Locally activated gated neural network for automatic music genre classification," *Applied Sciences*, vol. 13, no. 8, Art. no. 5010, 2023, doi: 10.3390/app13085010.',
    '[4] M. V. S. L. Jahnavi, A. Satapathy, C. Lokesh, and P. B. Likhitha, "A comparative performance evaluation of machine learning approaches for spectrogram-based music genre classification," in *Proc. 2023 IEEE 3rd Int. Conf. Technology, Engineering, Management for Societal Impact using Marketing, Entrepreneurship and Talent (TEMSMET)*, 2023, pp. 1–7, doi: 10.1109/TEMSMET56707.2023.10150176.',
    '[5] T. Li, "Optimizing the configuration of deep learning models for music genre classification," *Heliyon*, vol. 10, no. 2, Art. no. e24892, 2024, doi: 10.1016/j.heliyon.2024.e24892.',
    '[6] J. Shen and G. Xiao, "Music genre classification based on functional data analysis," *IEEE Access*, vol. 12, pp. 185482–185491, 2024, doi: 10.1109/ACCESS.2024.3512874.',
    '[7] Z. Zhao, Z. Xie, J. Fu, and X. Tian, "Music genre classification: Machine learning on GTZAN," *Applied and Computational Engineering*, vol. 79, pp. 219–233, 2024, doi: 10.54254/2755-2721/79/20241639.',
    '[8] D. E. Setyoko, C. G. S. Adhi, and R. Arisandi, "ResNet-18 with attention mechanism-Bidirectional LSTM hybrid approach for music genre classification using stacking MFCC and Mel-spectrogram features," *Indonesian Journal of Computer Science*, vol. 13, no. 6, 2024, doi: 10.33022/ijcs.v13i6.4456.',
    '[9] X. Deng, "Music genre classification and recognition using improved deep convolutional neural network-DenseNet-II," in *Proc. 2024 2nd Int. Conf. Data Science and Information System (ICDSIS)*, 2024, doi: 10.1109/ICDSIS61070.2024.10594180.',
    '[10] P. Wu, W. Gao, Y. Chen, F. Xu, Y. Ji, J. Tu, and H. Lin, "An improved ViT model for music genre classification based on Mel spectrogram," *PLOS ONE*, vol. 20, no. 3, Art. no. e0319027, 2025, doi: 10.1371/journal.pone.0319027.',
    '[11] S. Kundu, S. Panda, L. Pubbisetty, S. Palani, and P. N. Senthil Prakash, "Mel spectrogram based CNN with attention mechanisms for robust music genre classification," in *Proc. 2025 IEEE 9th Int. Conf. Information and Communication Technology (CICT)*, 2025, doi: 10.1109/CICT67193.2025.11398949.',
    '[12] M. Defferrard, K. Benzi, P. Vandergheynst, and X. Bresson, "FMA: A dataset for music analysis," in *Proc. 18th Int. Society for Music Information Retrieval Conf. (ISMIR)*, Suzhou, China, 2017, pp. 316–323.',
]


def section(source: str, heading: str) -> str:
    lines = source.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration as exc:
        raise ValueError(f"Expected draft section is missing: {heading}") from exc
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end]).strip()


def split_body(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    return lines[0], "\n".join(lines[1:]).strip()


def phase5_parameter_table(phase5: dict) -> str:
    rows = ["| Representation | Classifier | Inputs | Validation-selected settings |", "|---|---|---:|---|"]
    for rep in REP_ORDER:
        for model in MODEL_ORDER:
            item = phase5["selected"][f"{rep}__{model}"]
            params = item["selected_hyperparameters"]
            if model == "logistic_regression":
                setting = f"C={params['C']:g}"
            elif model == "svm":
                setting = f"C={params['C']:g}, gamma={params['gamma']}, 3-fold sigmoid calibration"
            elif model == "random_forest":
                depth = "None" if params["max_depth"] is None else str(params["max_depth"])
                setting = f"trees={params['n_estimators']}, max_depth={depth}"
            else:
                setting = f"layers={params['hidden_layers']}, lr={params['learning_rate']:g}, best epoch={item['best_epoch']}"
            rows.append(f"| {REP_LABEL[rep]} | {MODEL_LABEL[model]} | {item['input_features']} | `{setting}` |")
    return "\n".join(rows)


def phase5_result_table(csv: pd.DataFrame) -> str:
    rows = ["| Representation | Classifier | Features | Accuracy | Macro P | Macro R | Macro-F1 |",
            "|---|---|---:|---:|---:|---:|---:|"]
    subset = csv[csv.partition.eq("test")]
    for rep in REP_ORDER:
        for model in MODEL_ORDER:
            r = subset[(subset.representation == rep) & (subset.model == model)].iloc[0]
            rows.append(f"| {REP_LABEL[rep]} | {MODEL_LABEL[model]} | {int(r.input_features)} | {r.accuracy:.4f} | {r.macro_precision:.4f} | {r.macro_recall:.4f} | {r.macro_f1:.4f} |")
    return "\n".join(rows)


def phase5b_result_table(p5b: dict) -> str:
    rows = ["| Representation | Phase 5B selected C / gamma | Validation accuracy | Validation macro-F1 | Exploratory test accuracy* | Exploratory test macro-F1* |",
            "|---|---|---:|---:|---:|---:|"]
    for rep in REP_ORDER:
        val = p5b["selected"]["selected"][rep]
        test = p5b["test"][rep]
        hp = val["selected_hyperparameters"]
        rows.append(f"| {REP_LABEL[rep]} | {hp['C']:g} / {hp['gamma']} | {val['validation_metrics']['accuracy']:.4f} | {val['validation_metrics']['macro_f1']:.4f} | {test['accuracy']:.4f} | {test['macro_f1']:.4f} |")
    return "\n".join(rows)


def per_class_table(record: dict) -> str:
    rows = ["| Genre | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for genre in GENRES:
        m = record["per_class"][genre]
        rows.append(f"| {genre} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1-score']:.3f} | {int(m['support'])} |")
    return "\n".join(rows)


def majority_class_baseline(records: list[dict]) -> str:
    """Compute a constant majority-class reference from saved train/test counts."""
    features = pd.read_csv(ROOT / "data/features/fma_medium_features.csv",
                           usecols=["genre", "split"])
    train_counts = features.loc[features["split"].eq("training"), "genre"].value_counts()
    if train_counts.empty:
        raise ValueError("Cannot derive the majority-class baseline without training labels")
    majority = str(train_counts.index[0])
    test_record = next(
        r for r in records
        if r["partition"] == "test" and r["representation"] == "mutual_information"
        and r["model"] == "mlp"
    )
    supports = {g: int(test_record["per_class"][g]["support"]) for g in GENRES}
    total = sum(supports.values())
    if total != 1772 or set(supports) != set(GENRES):
        raise ValueError("Saved test supports do not match the verified 1,772 rows and 16 genres")
    correct = supports[majority]
    accuracy = correct / total
    macro_precision = accuracy / len(GENRES)
    macro_recall = 1 / len(GENRES)
    macro_f1 = (2 * correct / (total + correct)) / len(GENRES)
    return (
        f"**Majority-class reference.** A constant predictor that always returns the most "
        f"frequent training genre ({majority}; {int(train_counts.iloc[0]):,} training rows) "
        f"would score accuracy {accuracy:.4f}, macro precision {macro_precision:.4f}, macro "
        f"recall {macro_recall:.4f}, and macro-F1 {macro_f1:.4f} on the saved test class "
        f"supports. This is a descriptive reference calculated from saved train counts and "
        f"per-class test supports, not a fitted model or selection criterion."
    )


def get_source_sections(phase5: dict, evidence: dict, p5b: dict) -> str:
    source = DRAFT.read_text(encoding="utf-8")
    pieces = []

    abstract = section(source, "## Abstract")
    pieces.append(abstract)
    for source_heading, output_heading in [
        ("## I. Introduction and Motivation", "## 1. Introduction"),
        ("## II. Literature Survey", "## 2. Literature Review"),
        ("## III. Research Gap", "## 3. Research Gaps"),
        ("## IV. Problem Statement", "## 4. Problem Statement"),
        ("## V. Project Objectives", "## 5. Objectives"),
    ]:
        _, body = split_body(section(source, source_heading))
        pieces.append(output_heading + "\n\n" + body)

    _, dataset = split_body(section(source, "## VI. Dataset Description"))
    dataset_intro = (
        "The Free Music Archive dataset supports music information retrieval research [12]. "
        "The extracted `fma_medium` directory contains 25,000 audio files. Matching their IDs "
        "to the official metadata shows that 8,000 are assigned to FMA Small and 17,000 to "
        "FMA Medium. The project deliberately retains only the 17,000 metadata rows whose "
        "official `set/subset` value is `medium`; it does not add the Small tracks. All 17,000 "
        "retained rows have labels across 16 top-level genres and official splits. Feature extraction "
        "produced rows for 16,986 tracks and recorded 14 failures (12 decoder file-open errors "
        "and two decoder-internal errors). Those failures remain in the saved failure log and "
        "were not recovered or rerun."
    )
    dataset = re.sub(r"FMA Medium contains 17,000 official tracks[^.]*\.", dataset_intro, dataset, count=1)
    dataset = dataset.replace("The existing saved figures do not include a class-distribution chart, so this draft uses the split and class-support figures above instead.", "")
    methodology = section(source, "## VII. Proposed and Implemented Methodology")
    _, methodology = split_body(methodology)
    methodology = re.sub(r"^### A\. Audio preprocessing and feature extraction", "### 6.2 Audio Preprocessing and Feature Extraction", methodology, flags=re.M)
    methodology = re.sub(r"^### B\. Data partitions and leakage controls", "### 6.3 Data Partitions and Leakage Controls", methodology, flags=re.M)
    methodology = re.sub(r"^### C\. Feature representations", "### 6.4 Feature Representations", methodology, flags=re.M)
    methodology = re.sub(r"^### D\. Classifiers", "### 6.5 Classifiers", methodology, flags=re.M)
    methodology = re.sub(r"^### E\. Metrics and measurement", "### 6.6 Metrics and Measurement", methodology, flags=re.M)
    methodology = re.sub(r"^### F\. Experimental workflow", "### 6.7 Experimental Workflow", methodology, flags=re.M)
    methodology = methodology.replace(
        "### 6.5 Classifiers",
        "### 6.5 Classifiers\n\nThe project used seed 42. Logistic Regression used `max_iter=1000`; validation searched C in {0.1, 1, 10}. Phase 5 SVM used an RBF kernel with sigmoid calibration; validation searched C in {1, 10} and gamma in {`scale`, 0.01}. Random Forest validation searched 100 or 300 trees and maximum depth `None` or 20. The MLP used ReLU, dropout 0.30, Adam, cross-entropy loss, weight decay 0.0001, batch size 128, up to 100 epochs, and patience 12; validation searched hidden layers [256, 128] or [128, 64] and learning rate 0.001 or 0.0005.\n\n**Table 3. Phase 5 validation-selected settings**\n\n" + phase5_parameter_table(phase5),
    )
    methodology = methodology.replace(
        "PCA components are mixtures of original variables, not selected original features.",
        "PCA components are mixtures of original variables, not selected original features. "
        "The Mutual Information candidates retained 19, 37, or 56 features (ceilings of 25%, "
        "50%, and 75% of 74); Random Forest importance candidates retained 20, 40, or 60. "
        "The retained counts were selected using validation macro-F1. PCA used the configured "
        "95% variance threshold.",
    )
    pieces.append("## 6. Methodology\n\n### 6.1 Dataset and Data Collection\n\n" + dataset + "\n\n" + methodology)

    _, prediction = split_body(section(source, "## VIII. System Workflow and Prediction Interface"))
    prediction = prediction.replace("# Phase 5B — Conventional RBF SVC follow-up", "")
    prediction = prediction.replace("Phase 5B used a conventional RBF SVC with `probability=False` and no class weighting, fit on the full 13,510-row training partition.", "Phase 5B used a conventional RBF SVC with `probability=False`, fit on the full 13,510-row training partition.")
    prediction = prediction.replace(
        "python scripts\\predict_genre.py path\\to\\audio.mp3 --model models\\phase5b_svm\\phase5b_svm__mutual_information.joblib",
        "python scripts\\predict_genre.py path\\to\\audio.mp3 `\n  --model models\\phase5b_svm\\phase5b_svm__mutual_information.joblib",
    )
    prediction = prediction.replace("### Difference from original baseline", "### Phase 5 and Phase 5B SVM Procedures")
    pieces.append("### 6.8 Prediction Interface\n\n" + prediction)

    results = section(source, "## IX. Model Evaluation and Results")
    _, results = split_body(results)
    results = results.replace("../figures/", "results/figures/")
    for old, new in [
        ("### A. Phase 5 controlled 16-configuration comparison", "### 7.1 Phase 5 Main Experiment: 16 Configurations"),
        ("### B. Phase 5B conventional full-training-set SVC follow-up", "### 7.2 Phase 5B Conventional SVC Follow-Up"),
        ("### C. Genre-level results and confusion patterns", "### 7.3 Genre-Level Results and Confusion Patterns"),
        ("### D. Feature selection, PCA, and runtime trade-offs", "### 7.4 Feature Selection, PCA, and Runtime Trade-Offs"),
    ]:
        results = results.replace(old, new)
    results = re.sub(
        r"\| Representation \| Classifier \| Features \| Accuracy \| Macro P \| Macro R \| Macro-F1 \|\n(?:\|.*\n)+",
        phase5_result_table(evidence["csv"]) + "\n", results, count=1)
    results = re.sub(
        r"\| Representation \| Phase 5B selected C / gamma \| Validation accuracy \| Validation macro-F1 \| Phase 5B test accuracy\* \| Phase 5B test macro-F1\* \|\n(?:\|.*\n)+",
        phase5b_result_table(p5b) + "\n", results, count=1)
    results = re.sub(
        r"\| Genre \| Precision \| Recall \| F1 \| Support \|\n(?:\|.*\n)+",
        per_class_table(next(r for r in evidence["records"] if r["partition"] == "test" and r["representation"] == "mutual_information" and r["model"] == "mlp")) + "\n",
        results, count=1)
    results = results.replace(
        "Phase 5’s SVM was an RBF SVC wrapped in three-fold sigmoid `CalibratedClassifierCV`; its fitted bundle contains calibrated SVC members trained on CV-fold subsets and supports probabilities. Phase 5B used a conventional RBF SVC with `probability=False` and no class weighting, fit on the full 13,510-row training partition. Both pipelines scale features using training-fitted standardization. The Phase 5B grid was C = {0.1, 1, 10} and gamma = {`scale`, 0.001, 0.1}; nine candidates were compared per representation using validation macro-F1.",
        "Phase 5 used an RBF SVC with sigmoid calibration through `CalibratedClassifierCV` using three-fold cross-validation. Its saved calibrated bundle contains SVC members trained on CV-fold subsets and supports probability output. Phase 5B instead used a conventional RBF SVC with `probability=False`, fitted on all 13,510 training rows. Both approaches used standardization fitted on training data only. The Phase 5B grid compared C values {0.1, 1, 10} and gamma values {`scale`, 0.001, 0.1} per representation using validation macro-F1. Because calibration and training-row exposure differ, the score difference cannot be attributed to a single parameter.",
    )
    results = results.replace(
        "Across the validation results in the Phase 5 comparison,",
        majority_class_baseline(evidence["records"]) + "\n\nAcross the validation results in the Phase 5 comparison,",
        1,
    )
    results = results.replace(
        "The full representation had the highest average test macro-F1 across the four classifiers (0.3293),",
        "Across the four classifiers, mean test macro-F1 was 0.3293 for Full Features, 0.2967 for Mutual Information, 0.3187 for Random Forest Selection, and 0.2974 for PCA (95% Variance). Full Features was highest; this is a descriptive comparison on one split, not evidence of statistical significance. The full representation had the highest average test macro-F1 across the four classifiers (0.3293),",
        1,
    )
    results = re.sub(r"Existing figures:.*", "Figure 1 compares Phase 5 test accuracy; Figure 2 compares test macro-F1. Figures 3–6 show per-genre F1 by classifier. Figure 7 presents the confusion matrix for the Phase 5 validation-selected MI + MLP.", results)
    results = results.replace(
        "See [PCA explained variance](results/figures/pca_cumulative_variance.png) and [Random Forest feature importance](results/figures/random_forest_feature_importance.png).",
        "Figures 8 and 9 show the training-fitted PCA cumulative variance and Random Forest feature-importance ranking.",
    )
    # Figure markup is deliberately placed next to its discussion; the four per-model
    # heatmaps are generated from the saved Phase 5 JSON and contain the same values.
    figure_block = "\n\n".join([
        "![Figure 1. Test accuracy across the 16 finalized Phase 5 configurations; representation and classifier names are expanded for readability](tmp/final_report_assets/test_accuracy_comparison.png)",
        "![Figure 2. Test macro-F1 across the 16 finalized Phase 5 configurations; representation and classifier names are expanded for readability](tmp/final_report_assets/test_macro_f1_comparison.png)",
        "![Figure 3. Phase 5 per-genre F1 by representation for Logistic Regression](tmp/final_report_assets/heatmap_logistic_regression.png)",
        "![Figure 4. Phase 5 per-genre F1 by representation for calibrated SVM](tmp/final_report_assets/heatmap_svm.png)",
        "![Figure 5. Phase 5 per-genre F1 by representation for Random Forest](tmp/final_report_assets/heatmap_random_forest.png)",
        "![Figure 6. Phase 5 per-genre F1 by representation for PyTorch MLP](tmp/final_report_assets/heatmap_mlp.png)",
        "![Figure 7. Test confusion matrix counts for the validation-selected Phase 5 Mutual Information plus PyTorch MLP](tmp/final_report_assets/selected_mlp_confusion_matrix.png)",
    ])
    results = results.replace(
        "Figure 1 compares Phase 5 test accuracy; Figure 2 compares test macro-F1. Figures 3–6 show per-genre F1 by classifier. Figure 7 presents the confusion matrix for the Phase 5 validation-selected MI + MLP.",
        "Figure 1 compares Phase 5 test accuracy and Figure 2 compares test macro-F1. To make the class labels legible, Figures 3–6 separate the per-genre F1 heatmap by classifier. Figure 7 presents the confusion matrix for the Phase 5 validation-selected MI + MLP.\n\n" + figure_block,
    )
    results = results.replace(
        "Figures 8 and 9 show the training-fitted PCA cumulative variance and Random Forest feature-importance ranking.",
        "Figures 8 and 9 show the training-fitted PCA cumulative variance and Random Forest feature-importance ranking.\n\n"
        "![Figure 8. Cumulative explained variance from training-fitted PCA](results/figures/pca_cumulative_variance.png)\n\n"
        "![Figure 9. Top 25 training-derived Random Forest feature-importance scores](results/figures/random_forest_feature_importance.png)",
    )
    pieces.append("## 7. Results\n\n" + results)

    _, expected = split_body(section(source, "## X. Expected and Observed Outcomes"))
    _, significance = split_body(section(source, "## XI. Significance"))
    _, limitations = split_body(section(source, "## XII. Limitations"))
    discussion = (
        "## 8. Inference and Discussion\n\n"
        "### 8.1 Classifier and Representation Trade-Offs\n\n" + expected + "\n\n" + significance +
        "\n\n### 8.2 Class-Level Error Patterns and Limitations\n\n" + limitations
    )
    pieces.append(discussion)

    _, future = split_body(section(source, "## XIII. Future Scope"))
    conclusion = (
        "## 9. Conclusion and Future Work\n\n"
        "The project implemented an end-to-end feature-based FMA Medium genre classifier and compared the original 16 Phase 5 configurations under the shared official split protocol. Mutual Information plus PyTorch MLP was selected using validation macro-F1 (0.3313); its test accuracy was 0.6580 and test macro-F1 was 0.3280. Feature selection reduced input dimensionality, and PCA retained 95.2995% of training variance in 45 components, but neither representation change uniformly improved classification.\n\n"
        "The Phase 5B conventional MI SVC achieved validation macro-F1 0.3828 in a later, separate SVC-only follow-up. Its test macro-F1 of 0.3150 is exploratory because Phase 5 test results had already been inspected. It does not replace the original Phase 5 cross-family selection. The CLI uses the saved ordered MI features and training-fitted scaler with the conventional SVC to predict one of 16 labels; it does not return calibrated probabilities. No spectrogram-based CNN was implemented or evaluated.\n\n"
        "### 9.1 Future Work\n\n" + future
    )
    pieces.append(conclusion)
    pieces.append("## 10. References\n\n" + "\n\n".join(REFERENCES))
    return "\n\n".join(pieces)


def read_evidence() -> tuple[pd.DataFrame, dict, dict, dict]:
    needed = [DRAFT, ROOT / "results/metrics/experiment_results.csv",
              ROOT / "results/metrics/experiment_results.json",
              ROOT / "results/metrics/phase5_checkpoint.json",
              ROOT / "results/phase5b_svm/metrics/selected_configs.json",
              ROOT / "results/phase5b_svm/metrics/exploratory_test_metrics.json"]
    absent = [str(p.relative_to(ROOT)) for p in needed if not p.is_file()]
    if absent:
        raise FileNotFoundError(f"Required saved report evidence is missing: {absent}")
    csv = pd.read_csv(needed[1])
    records = json.loads(needed[2].read_text(encoding="utf-8"))
    phase5 = json.loads(needed[3].read_text(encoding="utf-8"))
    p5b = json.loads(needed[4].read_text(encoding="utf-8"))
    p5b_test = json.loads(needed[5].read_text(encoding="utf-8"))
    if csv.partition.value_counts().to_dict() != {"validation": 16, "test": 16}:
        raise ValueError("Saved Phase 5 table must contain all 16 validation and 16 test rows")
    winner = max(phase5["selected"].items(), key=lambda item: item[1]["validation_metrics"]["macro_f1"])
    if winner[0] != "mutual_information__mlp":
        raise ValueError(f"Unexpected Phase 5 validation winner: {winner[0]}")
    wm = winner[1]["validation_metrics"]
    if abs(wm["macro_f1"] - .331310) > 1e-6 or abs(wm["accuracy"] - .691901) > 1e-6:
        raise ValueError("Phase 5 validation-selected MI + MLP metrics differ from verified values")
    test = csv[(csv.partition == "test") & (csv.representation == "mutual_information") & (csv.model == "mlp")].iloc[0]
    if abs(test.accuracy - .658014) > 1e-6 or abs(test.macro_f1 - .328044) > 1e-6:
        raise ValueError("Phase 5 selected test metrics differ from the saved results")
    mi = p5b["selected"]["mutual_information"]
    if mi["selected_hyperparameters"] != {"C": 10.0, "gamma": "scale"} or abs(mi["validation_metrics"]["macro_f1"] - .3828451379993131) > 1e-12:
        raise ValueError("Phase 5B selected MI SVC does not match the saved validation results")
    if abs(p5b_test["mutual_information"]["macro_f1"] - .31503382601951796) > 1e-12:
        raise ValueError("Phase 5B exploratory MI SVC test metric differs from the saved result")
    if len(records) != 32:
        raise ValueError(f"Expected 32 saved Phase 5 metric records, found {len(records)}")
    return csv, {"records": records, "csv": csv}, phase5, {"selected": p5b, "test": p5b_test}


def make_heatmaps(records: list[dict]) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = [r for r in records if r.get("partition") == "test"]
    font_dir = Path(r"C:\Windows\Fonts")
    regular = ImageFont.truetype(str(font_dir / "arial.ttf"), 27)
    bold = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 32)
    small = ImageFont.truetype(str(font_dir / "arial.ttf"), 23)
    rep_font = ImageFont.truetype(str(font_dir / "arial.ttf"), 22)
    stops = [(0.0, (255, 255, 217)), (0.25, (199, 233, 180)),
             (0.5, (65, 182, 196)), (0.75, (34, 94, 168)), (1.0, (8, 29, 88))]

    def color_at(value: float) -> tuple[int, int, int]:
        for ix in range(len(stops) - 1):
            lo, c0 = stops[ix]
            hi, c1 = stops[ix + 1]
            if value <= hi:
                t = (value - lo) / (hi - lo)
                return tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))
        return stops[-1][1]

    by_key = {(r["model"], r["representation"]): r for r in rows}
    width, height = 1400, 1700
    left, right, top, bottom = 340, 135, 160, 180
    plot_w, plot_h = width - left - right, height - top - bottom
    cell_w, cell_h = plot_w / 4, plot_h / len(GENRES)
    rep_names = ["Full Features\n(74)", "Mutual Information\n(37)",
                 "Random Forest\nSelection (40)", "PCA (95% Variance)\n(45)"]
    for model in MODEL_ORDER:
        image = PILImage.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        title = f"Phase 5 test per-genre F1 — {MODEL_LABEL[model]}"
        box = draw.textbbox((0, 0), title, font=bold)
        draw.text(((width - (box[2] - box[0])) / 2, 35), title, font=bold, fill=(31, 54, 79))
        for col, (rep, rep_name) in enumerate(zip(REP_ORDER, rep_names)):
            label_box = draw.multiline_textbbox((0, 0), rep_name, font=rep_font, spacing=2, align="center")
            draw.multiline_text((left + col * cell_w + (cell_w - (label_box[2] - label_box[0])) / 2,
                                 top - 57), rep_name, font=rep_font, spacing=2, align="center",
                                fill=(38, 48, 58))
        for row, genre in enumerate(GENRES):
            y0 = top + row * cell_h
            label_box = draw.textbbox((0, 0), genre, font=regular)
            draw.text((left - 18 - (label_box[2] - label_box[0]), y0 + (cell_h - (label_box[3] - label_box[1])) / 2 - 3),
                      genre, font=regular, fill=(38, 48, 58))
            for col, rep in enumerate(REP_ORDER):
                value = float(by_key[(model, rep)]["per_class"][genre]["f1-score"])
                x0 = left + col * cell_w
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                fill = color_at(value)
                draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(255, 255, 255), width=3)
                fg = (255, 255, 255) if value >= .58 else (35, 45, 54)
                label = f"{value:.2f}"
                b = draw.textbbox((0, 0), label, font=regular)
                draw.text((x0 + (cell_w - (b[2] - b[0])) / 2,
                           y0 + (cell_h - (b[3] - b[1])) / 2 - 3), label, font=regular, fill=fg)
        draw.text((left + plot_w / 2 - 95, height - 72), "Feature representation", font=regular, fill=(38, 48, 58))
        # A compact vertical scale makes the shared 0–1 range explicit.
        bar_x0, bar_x1 = width - 86, width - 61
        for y in range(top, top + plot_h):
            v = 1 - (y - top) / plot_h
            draw.line((bar_x0, y, bar_x1, y), fill=color_at(v))
        for tick in [0, .25, .5, .75, 1.0]:
            y = top + plot_h * (1 - tick)
            draw.text((bar_x1 + 6, y - 14), f"{tick:.2f}", font=small, fill=(38, 48, 58))
        image.save(ASSETS / f"heatmap_{model}.png", optimize=True)


def make_report_comparison_charts(csv: pd.DataFrame) -> None:
    """Draw readable report-only charts from the saved Phase 5 test summary."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    font_dir = Path(r"C:\Windows\Fonts")
    regular = ImageFont.truetype(str(font_dir / "arial.ttf"), 27)
    bold = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 34)
    small = ImageFont.truetype(str(font_dir / "arial.ttf"), 22)
    colors_by_model = {
        "logistic_regression": (51, 102, 153), "svm": (224, 135, 45),
        "random_forest": (86, 150, 107), "mlp": (182, 76, 76),
    }
    names = ["Full\nFeatures", "Mutual\nInformation", "Random Forest\nSelection", "PCA (95%\nVariance)"]
    tests = csv.loc[csv.partition.eq("test")]
    for metric, title, filename in [
        ("accuracy", "Phase 5 Test Accuracy", "test_accuracy_comparison.png"),
        ("macro_f1", "Phase 5 Test Macro-F1", "test_macro_f1_comparison.png"),
    ]:
        width, height = 1600, 850
        image = PILImage.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        title_box = draw.textbbox((0, 0), title, font=bold)
        draw.text(((width - title_box[2] + title_box[0]) / 2, 24), title,
                  font=bold, fill=(31, 54, 79))
        left, right, top, bottom = 120, 50, 145, 175
        plot_w, plot_h = width - left - right, height - top - bottom
        for tick in range(0, 6):
            value = tick / 5
            y = top + plot_h * (1 - value)
            draw.line((left, y, left + plot_w, y), fill=(218, 225, 232), width=2)
            draw.text((left - 66, y - 14), f"{value:.1f}", font=small, fill=(55, 67, 79))
        score_label = PILImage.new("RGBA", (180, 38), (255, 255, 255, 0))
        ImageDraw.Draw(score_label).text((4, 2), "Score (0-1)", font=small, fill=(38, 48, 58))
        rotated_score = score_label.rotate(90, expand=True, resample=PILImage.Resampling.BICUBIC)
        image.paste(rotated_score, (24, round(top + plot_h / 2 - rotated_score.height / 2)), rotated_score)
        draw.line((left, top, left, top + plot_h), fill=(70, 82, 95), width=2)
        draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(70, 82, 95), width=2)
        group_w = plot_w / len(REP_ORDER)
        bar_w, gap = 45, 12
        total_bar_w = len(MODEL_ORDER) * bar_w + (len(MODEL_ORDER) - 1) * gap
        for rep_i, rep in enumerate(REP_ORDER):
            center = left + group_w * (rep_i + .5)
            start_x = center - total_bar_w / 2
            for model_i, model in enumerate(MODEL_ORDER):
                row = tests.loc[(tests.representation == rep) & (tests.model == model)]
                if len(row) != 1:
                    raise ValueError(f"Expected one saved test result for {rep}/{model}")
                score = float(row.iloc[0][metric])
                x0 = start_x + model_i * (bar_w + gap)
                y0 = top + plot_h * (1 - score)
                draw.rectangle((x0, y0, x0 + bar_w, top + plot_h),
                               fill=colors_by_model[model])
            box = draw.multiline_textbbox((0, 0), names[rep_i], font=regular, spacing=3, align="center")
            draw.multiline_text((center - (box[2] - box[0]) / 2, top + plot_h + 15), names[rep_i],
                                font=regular, spacing=3, align="center", fill=(38, 48, 58))
        legend_y = 93
        legend_x = 175
        for model in MODEL_ORDER:
            label = MODEL_LABEL[model]
            draw.rectangle((legend_x, legend_y, legend_x + 22, legend_y + 22), fill=colors_by_model[model])
            draw.text((legend_x + 30, legend_y - 4), label, font=small, fill=(38, 48, 58))
            legend_x += 340
        image.save(ASSETS / filename, optimize=True)


def make_report_confusion_matrix(record: dict) -> None:
    """Render saved confusion counts with annotations and readable class labels."""
    matrix = record["confusion_matrix"]
    if len(matrix) != len(GENRES) or any(len(row) != len(GENRES) for row in matrix):
        raise ValueError("Saved selected-model confusion matrix is not 16 by 16")
    if sum(sum(int(v) for v in row) for row in matrix) != 1772:
        raise ValueError("Saved selected-model confusion matrix does not sum to 1,772 test rows")
    font_dir = Path(r"C:\Windows\Fonts")
    regular = ImageFont.truetype(str(font_dir / "arial.ttf"), 24)
    bold = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 25)
    width, height = 1700, 1740
    left, top, plot = 400, 100, 1120
    cell = plot / len(GENRES)
    image = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    maximum = max(max(map(int, row)) for row in matrix)
    for i, genre in enumerate(GENRES):
        y = top + i * cell
        box = draw.textbbox((0, 0), genre, font=regular)
        draw.text((left - 18 - (box[2] - box[0]), y + (cell - (box[3] - box[1])) / 2 - 2),
                  genre, font=regular, fill=(38, 48, 58))
        for j, raw in enumerate(matrix[i]):
            value = int(raw)
            fraction = value / maximum if maximum else 0
            shade = round(248 - 165 * fraction)
            fill = (shade, min(255, shade + 4), min(255, shade + 9))
            x = left + j * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=fill,
                           outline=(255, 255, 255), width=2)
            if value:
                text = str(value)
                b = draw.textbbox((0, 0), text, font=bold)
                fg = (255, 255, 255) if fraction > .58 else (31, 45, 59)
                draw.text((x + (cell - (b[2] - b[0])) / 2,
                           y + (cell - (b[3] - b[1])) / 2 - 2), text, font=bold, fill=fg)
    x_labels = ["Blues", "Classical", "Country", "Easy Listening", "Electronic", "Experimental",
                "Folk", "Hip-Hop", "Instrumental", "International", "Jazz", "Old-Time / Historic",
                "Pop", "Rock", "Soul-RnB", "Spoken"]
    for j, label in enumerate(x_labels):
        label_img = PILImage.new("RGBA", (520, 36), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_img)
        label_draw.text((4, 2), label, font=regular, fill=(38, 48, 58))
        rotated = label_img.rotate(55, expand=True, resample=PILImage.Resampling.BICUBIC)
        cx = left + (j + .5) * cell
        image.paste(rotated, (round(cx - rotated.width / 2), top + plot + 8), rotated)
    draw = ImageDraw.Draw(image)
    draw.text((left + plot / 2 - 110, height - 35), "Predicted genre", font=regular, fill=(38, 48, 58))
    true_label = PILImage.new("RGBA", (170, 36), (255, 255, 255, 0))
    ImageDraw.Draw(true_label).text((4, 2), "True genre", font=regular, fill=(38, 48, 58))
    rotated_true = true_label.rotate(90, expand=True, resample=PILImage.Resampling.BICUBIC)
    image.paste(rotated_true, (18, round(top + plot / 2 - rotated_true.height / 2)), rotated_true)
    image.save(ASSETS / "selected_mlp_confusion_matrix.png", optimize=True)


def markdown_inline(text: str) -> str:
    text = escape(text, {'"': "&quot;"})
    text = re.sub(r"`([^`]+)`", r"<font name='ReportCode' size='8.6'>\1</font>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?<!\\)_([^_]+)_(?!\\)", r"<i>\1</i>", text)
    return text.replace("\\`", "`")


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        self._heading = None

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and getattr(flowable.style, "outlineLevel", None) is not None:
            level = flowable.style.outlineLevel
            text = flowable.getPlainText()
            key = f"h{level}-{self.page}-{len(text)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=False)
            # The cover is intentionally unnumbered, so the TOC uses printed page labels.
            self.notify("TOCEntry", (level, text, max(1, self.page - 1), key))


def add_page_decor(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#C7D1DC"))
    canvas.setLineWidth(.45)
    canvas.line(20 * mm, A4[1] - 14 * mm, A4[0] - 20 * mm, A4[1] - 14 * mm)
    canvas.setFont("ReportBody", 8.2)
    canvas.setFillColor(colors.HexColor("#536273"))
    canvas.drawString(20 * mm, A4[1] - 11 * mm, SHORT_TITLE)
    canvas.line(20 * mm, 14 * mm, A4[0] - 20 * mm, 14 * mm)
    canvas.drawCentredString(A4[0] / 2, 9 * mm, str(doc.page - 1))
    canvas.restoreState()


def cover_decor(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#244B70"))
    canvas.rect(24 * mm, A4[1] - 25 * mm, A4[0] - 48 * mm, 2.2 * mm, fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor("#D5DEE7"))
    canvas.line(30 * mm, 30 * mm, A4[0] - 30 * mm, 30 * mm)
    canvas.restoreState()


def reference_para(text: str, styles):
    return Paragraph(markdown_inline(text), styles["Reference"])


def build_pdf(markdown: str, output: Path) -> None:
    fontdir = Path(r"C:\Windows\Fonts")
    pdfmetrics.registerFont(TTFont("ReportBody", str(fontdir / "times.ttf")))
    pdfmetrics.registerFont(TTFont("ReportBold", str(fontdir / "timesbd.ttf")))
    pdfmetrics.registerFont(TTFont("ReportItalic", str(fontdir / "timesi.ttf")))
    pdfmetrics.registerFont(TTFont("ReportCode", str(fontdir / "consola.ttf")))
    pdfmetrics.registerFontFamily("ReportBody", normal="ReportBody", bold="ReportBold", italic="ReportItalic", boldItalic="ReportBold")

    left = right = 20 * mm
    top, bottom = 20 * mm, 20 * mm
    frame = Frame(left, bottom, A4[0] - left - right, A4[1] - top - bottom,
                  id="normal", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc = ReportDocTemplate(str(output), pagesize=A4, leftMargin=left, rightMargin=right,
                            topMargin=top, bottomMargin=bottom, title=TITLE,
                            author="Automatic Music Genre Classification project group",
                            subject="FMA Medium acoustic-feature classification study")
    doc.addPageTemplates([PageTemplate(id="Cover", frames=frame, onPage=cover_decor),
                          PageTemplate(id="Body", frames=frame, onPage=add_page_decor)])
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportText", fontName="ReportBody", fontSize=10.1, leading=14.2,
        alignment=TA_JUSTIFY, spaceAfter=6.5, allowWidows=0, allowOrphans=0))
    styles.add(ParagraphStyle(name="ReportH1", fontName="ReportBold", fontSize=15.3, leading=19,
        textColor=colors.HexColor("#142D4E"), spaceBefore=13, spaceAfter=7, keepWithNext=True, outlineLevel=0))
    styles.add(ParagraphStyle(name="ReportH2", fontName="ReportBold", fontSize=11.7, leading=14.5,
        textColor=colors.HexColor("#244B70"), spaceBefore=9, spaceAfter=5, keepWithNext=True, outlineLevel=1))
    styles.add(ParagraphStyle(name="ReportH3", fontName="ReportBold", fontSize=10.8, leading=13.4,
        textColor=colors.HexColor("#244B70"), spaceBefore=7, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle(name="TableBody", fontName="ReportBody", fontSize=8.1, leading=10,
        alignment=TA_LEFT, spaceAfter=0))
    styles.add(ParagraphStyle(name="TableHead", fontName="ReportBold", fontSize=8.0, leading=9.6,
        alignment=TA_CENTER, textColor=colors.white, spaceAfter=0))
    styles.add(ParagraphStyle(name="FigureCaption", fontName="ReportBody", fontSize=8.8, leading=11.2,
        alignment=TA_LEFT, spaceBefore=3, spaceAfter=9, keepWithNext=False))
    styles.add(ParagraphStyle(name="TOCHead", fontName="ReportBold", fontSize=20, leading=25,
        textColor=colors.HexColor("#142D4E"), spaceAfter=16))
    styles.add(ParagraphStyle(name="TOC0", fontName="ReportBody", fontSize=10, leading=14,
        leftIndent=0, firstLineIndent=0, spaceBefore=2))
    styles.add(ParagraphStyle(name="TOC1", fontName="ReportBody", fontSize=9.2, leading=12.2,
        leftIndent=15, firstLineIndent=0))
    styles.add(ParagraphStyle(name="CoverTitle", fontName="ReportBold", fontSize=21.5, leading=27,
        textColor=colors.HexColor("#142D4E"), alignment=TA_CENTER, spaceAfter=13))
    styles.add(ParagraphStyle(name="CoverSub", fontName="ReportBody", fontSize=13, leading=18,
        textColor=colors.HexColor("#3F566E"), alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="CoverDescriptor", fontName="ReportItalic", fontSize=11, leading=15,
        textColor=colors.HexColor("#536273"), alignment=TA_CENTER, spaceAfter=7))
    styles.add(ParagraphStyle(name="Member", fontName="ReportBody", fontSize=11, leading=16,
        alignment=TA_LEFT, spaceAfter=0))
    styles.add(ParagraphStyle(name="CodeBlock", fontName="ReportCode", fontSize=8.7, leading=11.2,
        leftIndent=10, rightIndent=8, borderColor=colors.HexColor("#D5DEE7"), borderWidth=.5,
        borderPadding=7, backColor=colors.HexColor("#F5F7FA"), spaceBefore=3, spaceAfter=9))
    styles.add(ParagraphStyle(name="Reference", fontName="ReportBody", fontSize=9, leading=12.2,
        alignment=TA_LEFT, leftIndent=17, firstLineIndent=-17, spaceAfter=6.4, allowWidows=0, allowOrphans=0))
    styles.add(ParagraphStyle(name="Callout", fontName="ReportItalic", fontSize=9.1, leading=12.5,
        leftIndent=9, rightIndent=8, borderColor=colors.HexColor("#A9BBCD"), borderWidth=.5,
        borderPadding=7, backColor=colors.HexColor("#F4F7FA"), spaceBefore=3, spaceAfter=8))

    page_w = A4[0] - left - right
    story = [Spacer(1, 38 * mm), Paragraph(TITLE, styles["CoverTitle"]),
             Paragraph(SUBTITLE, styles["CoverSub"]), Spacer(1, 9 * mm),
             Paragraph("Academic Mini-Project Report", styles["CoverDescriptor"]),
             Spacer(1, 16 * mm), Paragraph("Prepared by", ParagraphStyle("Prepared", parent=styles["CoverDescriptor"]))]
    member_rows = [[Paragraph("Atharv Sethi", styles["Member"]), Paragraph("Registration No. 240968228", styles["Member"])],
                   [Paragraph("Kartik Pathak", styles["Member"]), Paragraph("Registration No. 240968152", styles["Member"])],
                   [Paragraph("Nalla Avaneesh Reddy", styles["Member"]), Paragraph("Registration No. 240968104", styles["Member"])]]
    members = Table(member_rows, colWidths=[page_w * .48, page_w * .52], hAlign="CENTER")
    members.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 18), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                                 ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                                 ("LINEBELOW", (0, 0), (-1, -2), .35, colors.HexColor("#E3E8EE"))]))
    story.extend([members, NextPageTemplate("Body"), PageBreak(), Paragraph("Contents", styles["TOCHead"])])
    toc = TableOfContents()
    toc.levelStyles = [styles["TOC0"], styles["TOC1"]]
    toc.dotsMinLevel = 0
    story.extend([toc, PageBreak()])

    lines = markdown.splitlines()
    i = 0
    para: list[str] = []

    def flush_para():
        if not para:
            return
        raw = " ".join(p.strip() for p in para)
        text = markdown_inline(raw)
        if raw.startswith("> "):
            story.append(Paragraph(text[2:] if text.startswith("&gt; ") else text, styles["Callout"]))
        elif re.match(r"^\d+\.\s+", raw):
            story.append(Paragraph(text, ParagraphStyle("Numbered", parent=styles["ReportText"], leftIndent=14, firstLineIndent=-14, spaceAfter=3)))
        elif raw.startswith("- "):
            story.append(Paragraph("–  " + text[2:], ParagraphStyle("Bullet", parent=styles["ReportText"], leftIndent=14, firstLineIndent=-9, spaceAfter=3)))
        elif raw.startswith("[1]") or re.match(r"^\[\d+\]", raw):
            story.append(reference_para(raw, styles))
        else:
            story.append(Paragraph(text, styles["ReportText"]))
        para.clear()

    def add_table(raw_rows: list[str]):
        content = []
        for row_i, row in enumerate(raw_rows):
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells):
                continue
            sty = styles["TableHead"] if not content else styles["TableBody"]
            content.append([Paragraph(markdown_inline(cell), sty) for cell in cells])
        if not content:
            return
        n = len(content[0])
        if n == 7:
            fractions = [.17, .22, .08, .13, .13, .13, .14]
        elif n == 6:
            fractions = [.24, .17, .145, .145, .15, .15]
        elif n == 5:
            fractions = [.28, .16, .17, .17, .22]
        elif n == 4 and "Validation-selected settings" in raw_rows[0]:
            fractions = [.17, .20, .09, .54]
        elif n == 3 and "Feature family" in raw_rows[0]:
            fractions = [.35, .46, .19]
        elif n == 4:
            fractions = [.25, .25, .29, .21]
        else:
            fractions = [1 / n] * n
        widths = [page_w * f for f in fractions]
        table = Table(content, colWidths=widths, repeatRows=1, splitByRow=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244B70")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#C8D0D9")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F8")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        block = [table, Spacer(1, 7)]
        story.append(KeepTogether(block) if len(content) <= 20 else table)
        if len(content) > 20:
            story.append(Spacer(1, 7))

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush_para(); i += 1; continue
        if line.startswith("```"):
            flush_para(); i += 1; code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i]); i += 1
            if i < len(lines): i += 1
            story.append(Preformatted("\n".join(code), styles["CodeBlock"], maxLineLength=95))
            continue
        if line.startswith("|"):
            flush_para(); table_rows = []
            while i < len(lines) and lines[i].startswith("|"):
                table_rows.append(lines[i]); i += 1
            add_table(table_rows); continue
        if line.startswith("!["):
            flush_para()
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                caption, rel = match.groups()
                img_path = (ROOT / rel).resolve()
                if not img_path.is_file():
                    raise FileNotFoundError(f"Referenced report figure is missing: {img_path}")
                img = Image(str(img_path))
                scale = min(page_w / img.imageWidth, (8.5 * inch) / img.imageHeight, 1.0)
                img.drawWidth = img.imageWidth * scale
                img.drawHeight = img.imageHeight * scale
                story.append(KeepTogether([img, Paragraph(markdown_inline(caption), styles["FigureCaption"])]))
            i += 1; continue
        if line.startswith("# "):
            flush_para(); i += 1; continue
        if line.startswith("## "):
            flush_para(); story.append(Paragraph(markdown_inline(line[3:]), styles["ReportH1"])); i += 1; continue
        if line.startswith("### "):
            flush_para(); story.append(Paragraph(markdown_inline(line[4:]), styles["ReportH2"])); i += 1; continue
        if line.startswith("#### "):
            flush_para(); story.append(Paragraph(markdown_inline(line[5:]), styles["ReportH3"])); i += 1; continue
        if line.startswith("- ") or re.match(r"^\d+\.\s+", line):
            flush_para()
            para.append(line)
            flush_para()
            i += 1
            continue
        para.append(line)
        i += 1
    flush_para()
    doc.multiBuild(story)


def verify_pdf(path: Path) -> tuple[int, str]:
    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if not 10 <= len(pdf.pages) <= 30:
            raise ValueError(f"Unexpected report page count: {len(pdf.pages)}")
    lowered = text.lower()
    forbidden = ["phase 8e", "phase8e", "0.3879", "0.6191", "0.0051", "class-weighting follow-up"]
    found = [item for item in forbidden if item in lowered]
    if found:
        raise ValueError(f"Disallowed Phase 8E material found in PDF text: {found}")
    normalized = " ".join(text.split())
    if TITLE not in normalized or "FMA Medium Dataset" not in normalized:
        raise ValueError("The PDF title page is missing the approved title or subtitle")
    for value in ["0.3313", "0.6919", "0.6580", "0.3280", "0.3828", "0.3150"]:
        if value not in text:
            raise ValueError(f"Required verified Phase 5/5B metric is absent from PDF text: {value}")
    if "exploratory" not in lowered or "unbiased final estimates" not in lowered:
        raise ValueError("Phase 5B test results are missing their exploratory caveat")
    for label in ["1. Introduction", "2. Literature Review", "3. Research Gaps", "4. Problem Statement",
                  "5. Objectives", "6. Methodology", "7. Results", "8. Inference and Discussion",
                  "9. Conclusion and Future Work", "10. References"]:
        if label not in text:
            raise ValueError(f"Expected report section not found: {label}")
    if "�" in text:
        raise ValueError("PDF text contains a replacement glyph; inspect font coverage")
    missing_figures = [i for i in range(1, 10) if f"Figure {i}." not in text]
    if missing_figures:
        raise ValueError(f"Expected figure captions are missing: {missing_figures}")
    for phrase in ["25,000 audio files", "8,000 are assigned to FMA Small",
                   "17,000 metadata rows", "16,986 tracks", "13,510", "1,704", "1,772",
                   "0.3442", "0.0215", "0.0625", "0.0320", "0.3293", "0.2967",
                   "0.3187", "0.2974"]:
        if phrase not in normalized:
            raise ValueError(f"Required dataset or verified comparison value is missing: {phrase}")
    if "does not add the small tracks" not in normalized.lower():
        raise ValueError("The report does not explain the intentional FMA Small exclusion")
    return len(pdf.pages), text


def main() -> None:
    if not DRAFT.is_file() or not OUT.is_file():
        raise FileNotFoundError("Expected reviewed Markdown draft and existing PDF in results/reports")
    _, evidence, phase5, p5b = read_evidence()
    markdown = get_source_sections(phase5, evidence, p5b)
    make_heatmaps(evidence["records"])
    make_report_comparison_charts(evidence["csv"])
    selected_mlp = next(
        r for r in evidence["records"]
        if r["partition"] == "test" and r["representation"] == "mutual_information"
        and r["model"] == "mlp"
    )
    make_report_confusion_matrix(selected_mlp)
    temp_pdf = OUT.with_name(".final_report_build.tmp.pdf")
    try:
        build_pdf(markdown, temp_pdf)
        pages, extracted = verify_pdf(temp_pdf)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = OUT.with_name(f"final_report_backup_pre_polish_{timestamp}.pdf")
        shutil.copy2(OUT, backup)
        os.replace(temp_pdf, OUT)
        final_pages, _ = verify_pdf(OUT)
        if pages != final_pages:
            raise ValueError("Final PDF changed during replacement verification")
        print(f"PDF: {OUT}")
        print(f"Backup: {backup}")
        print(f"Pages: {final_pages}; bytes: {OUT.stat().st_size}")
        print("Verified Phase 5/5B metrics, title, sections, exploratory caveat, and absence of Phase 8E content.")
    finally:
        if temp_pdf.exists():
            temp_pdf.unlink()
        if ASSETS.exists():
            shutil.rmtree(ASSETS)


if __name__ == "__main__":
    main()
