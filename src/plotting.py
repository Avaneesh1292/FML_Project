"""Publication-ready plots from saved metadata and experiment summaries."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_class_distribution(metadata: pd.DataFrame, output: str | Path) -> None:
    counts = metadata["genre"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    counts.plot.barh(ax=ax, color="#2878B5")
    ax.set(title="FMA Medium class distribution", xlabel="Tracks", ylabel="Top-level genre")
    fig.tight_layout()
    _save(fig, output)


def plot_feature_importance(ranking: pd.Series, output: str | Path, top_n: int = 25,
                            title: str = "Random Forest feature importance") -> None:
    selected = ranking.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(selected))))
    selected.plot.barh(ax=ax, color="#4C956C")
    ax.set(title=title, xlabel="Importance", ylabel="Feature")
    fig.tight_layout()
    _save(fig, output)


def plot_pca_variance(pca: Any, output: str | Path) -> None:
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, color="#2878B5")
    ax.axhline(0.95, color="#C44E52", linestyle="--", label="95% threshold")
    ax.set(xlabel="Principal components", ylabel="Cumulative explained variance",
           ylim=(0, 1.02), title="PCA cumulative explained variance")
    ax.legend()
    fig.tight_layout()
    _save(fig, output)


def plot_metric_comparison(results: pd.DataFrame, metric: str, output: str | Path) -> None:
    frame = results.loc[results["partition"].eq("test")]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=frame, x="representation", y=metric, hue="model", ax=ax)
    ax.set(title=f"Held-out test {metric.replace('_', ' ').title()} comparison", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    _save(fig, output)


def plot_confusion(matrix: list[list[int]], classes: list[str], output: str | Path,
                   title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(np.asarray(matrix), cmap="Blues", annot=False, xticklabels=classes,
                yticklabels=classes, ax=ax)
    ax.set(title=title, xlabel="Predicted genre", ylabel="True genre")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save(fig, output)


def _save(fig, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

