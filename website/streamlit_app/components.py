"""Reusable Streamlit rendering components for the genre classifier UI.

Every function in this module either returns a Plotly figure or calls
``st.markdown()`` / other Streamlit APIs to render HTML directly.  They
are pure-display helpers with **no** ML logic.
"""
from __future__ import annotations

from collections import Counter
from html import escape
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from web_utils import format_duration, format_file_size


# ── Hero ──────────────────────────────────────────────────────────────────────


def render_page_header() -> None:
    st.markdown(
        """
        <div class="app-heading">
            <div class="app-kicker">FML MINI PROJECT · FMA MEDIUM</div>
            <h1>Music Genre Classification</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Section headers ──────────────────────────────────────────────────────────


def render_section_header(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header"><h2>{escape(title)}</h2></div>',
        unsafe_allow_html=True,
    )


# ── Audio info ────────────────────────────────────────────────────────────────


def render_audio_info(
    filename: str,
    filesize: int,
    duration: float,
    sr: int,
    channels: int,
    original_sr: int,
) -> None:
    ch_label = "Mono" if channels == 1 else f"{channels} ch"
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="color:#c084fc;font-weight:700;font-size:1rem;margin-bottom:10px">
                {escape(filename)}
            </div>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-item-label">Duration</div>
                    <div class="info-item-value">{format_duration(duration)}</div>
                </div>
                <div class="info-item">
                    <div class="info-item-label">File Size</div>
                    <div class="info-item-value">{format_file_size(filesize)}</div>
                </div>
                <div class="info-item">
                    <div class="info-item-label">Sample Rate</div>
                    <div class="info-item-value">{original_sr:,} Hz</div>
                </div>
                <div class="info-item">
                    <div class="info-item-label">Channels</div>
                    <div class="info-item-value">{ch_label}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Waveform chart ────────────────────────────────────────────────────────────


def create_waveform_figure(
    times: np.ndarray,
    amplitudes: np.ndarray,
    start_sec: float,
    end_sec: float,
    total_duration: float,
) -> go.Figure:
    """Return a Plotly figure showing the waveform with a highlighted segment."""
    fig = go.Figure()

    # Waveform envelope
    fig.add_trace(
        go.Scatter(
            x=times,
            y=amplitudes,
            mode="lines",
            line=dict(color="rgba(139, 92, 246, 0.65)", width=1),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.10)",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Highlighted 30-second selection
    fig.add_vrect(
        x0=start_sec,
        x1=end_sec,
        fillcolor="rgba(124, 58, 237, 0.18)",
        line=dict(color="rgba(124, 58, 237, 0.70)", width=2, dash="dot"),
        layer="above",
        annotation_text="Selected segment",
        annotation_position="top left",
        annotation=dict(font_size=11, font_color="#a78bfa"),
    )

    # Start / end markers
    for t, label in [(start_sec, "Start"), (end_sec, "End")]:
        fig.add_vline(
            x=t,
            line=dict(color="#7c3aed", width=1.5),
        )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=40),
        height=180,
        xaxis=dict(
            title=dict(text="Time (seconds)", font=dict(color="#64748b", size=11)),
            tickfont=dict(color="#64748b", size=10),
            gridcolor="rgba(255,255,255,0.03)",
            range=[0, total_duration],
            zeroline=False,
        ),
        yaxis=dict(
            visible=False,
            range=[float(amplitudes.min()) * 1.15, float(amplitudes.max()) * 1.15],
        ),
        dragmode=False,
        showlegend=False,
    )
    return fig


# ── Segment indicator ─────────────────────────────────────────────────────────


def render_segment_indicator(start_sec: float, end_sec: float) -> None:
    st.markdown(
        f"""
        <div class="segment-indicator">
            <span class="segment-time">{format_duration(start_sec)}</span>
            <span class="segment-arrow">→</span>
            <span class="segment-time">{format_duration(end_sec)}</span>
            <span class="segment-duration-label">&nbsp;(30 s)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Prediction result card ────────────────────────────────────────────────────


def render_prediction_card(result: dict[str, Any]) -> None:
    start = result["start"]
    end = result["end"]
    st.markdown(
        f"""
        <div class="prediction-card">
            <div style="font-size:0.82rem;color:#64748b;text-transform:uppercase;
                        letter-spacing:0.08em;font-weight:600">Predicted Genre</div>
            <div class="genre-label">{escape(str(result['genre']))}</div>
            <div class="prediction-meta">
                {escape(str(result['filename']))} &ensp;·&ensp;
                {format_duration(start)} – {format_duration(end)}
            </div>
            <div class="prediction-model">Model: {escape(str(result.get('model_name', 'RBF-SVM')))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Metric cards ──────────────────────────────────────────────────────────────


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Comparison table ──────────────────────────────────────────────────────────


def render_comparison_table(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    rows = ""
    for r in results:
        rows += (
            f"<tr>"
            f"<td style='text-align:center'>#{r['segment_number']}</td>"
            f"<td>{format_duration(r['start'])} – {format_duration(r['end'])}</td>"
            f"<td style='color:#c084fc;font-weight:600'>{r['genre']}</td>"
            f"<td>{escape(str(r.get('model_name', 'RBF-SVM')))}</td>"
            f"</tr>"
        )
    st.markdown(
        f"""
        <table class="comparison-table">
            <thead><tr>
                <th style="text-align:center">Seg</th>
                <th>Time Range</th>
                <th>Predicted Genre</th>
                <th>Model</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


# ── Charts ────────────────────────────────────────────────────────────────────


def create_genre_distribution_chart(results: list[dict[str, Any]]) -> go.Figure:
    """Bar chart of predicted genre counts."""
    genres = [r["genre"] for r in results]
    counts = Counter(genres)
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels = [g for g, _ in sorted_items]
    values = [c for _, c in sorted_items]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=values,
                colorscale=[[0, "#6366f1"], [0.5, "#7c3aed"], [1, "#c084fc"]],
                line=dict(width=0),
            ),
            text=values,
            textposition="outside",
            textfont=dict(color="#c084fc", size=13, family="Inter"),
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=40),
        height=320,
        xaxis=dict(
            tickfont=dict(color="#94a3b8", size=11),
            gridcolor="rgba(255,255,255,0.03)",
        ),
        yaxis=dict(
            tickfont=dict(color="#64748b", size=10),
            gridcolor="rgba(255,255,255,0.05)",
            title=dict(text="Segments", font=dict(color="#64748b", size=11)),
        ),
        bargap=0.35,
    )
    return fig


def create_prediction_timeline(results: list[dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart showing predicted genre per segment in time order."""
    # Sort by start time
    sorted_res = sorted(results, key=lambda r: r["start"])

    labels = [f"#{r['segment_number']}" for r in sorted_res]
    starts = [r["start"] for r in sorted_res]
    durations = [r["end"] - r["start"] for r in sorted_res]
    genres = [r["genre"] for r in sorted_res]
    hovers = [
        f"Segment #{r['segment_number']}<br>"
        f"{format_duration(r['start'])} – {format_duration(r['end'])}<br>"
        f"Genre: {r['genre']}"
        for r in sorted_res
    ]

    # Assign a colour per unique genre
    unique_genres = list(dict.fromkeys(genres))  # preserve order
    palette = [
        "#7c3aed", "#6366f1", "#8b5cf6", "#a78bfa", "#c084fc",
        "#818cf8", "#6d28d9", "#4f46e5", "#7e22ce", "#9333ea",
        "#a855f7", "#d946ef", "#5b21b6", "#4338ca", "#3730a3", "#312e81",
    ]
    genre_color = {g: palette[i % len(palette)] for i, g in enumerate(unique_genres)}
    colors = [genre_color[g] for g in genres]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=durations,
            base=starts,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=genres,
            textposition="inside",
            textfont=dict(color="white", size=11, family="Inter"),
            hovertext=hovers,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=10, t=10, b=40),
        height=max(180, len(results) * 44 + 60),
        xaxis=dict(
            title=dict(text="Time (seconds)", font=dict(color="#64748b", size=11)),
            tickfont=dict(color="#64748b", size=10),
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            tickfont=dict(color="#94a3b8", size=11),
            autorange="reversed",
        ),
        showlegend=False,
    )
    return fig


# ── Analysis summary ──────────────────────────────────────────────────────────


def render_analysis_summary(results: list[dict[str, Any]]) -> None:
    genres = [r["genre"] for r in results]
    counts = Counter(genres)
    most_common = counts.most_common(1)[0][0] if counts else "—"
    cols = st.columns(3)
    with cols[0]:
        render_metric_card("Segments Analyzed", str(len(results)))
    with cols[1]:
        render_metric_card("Distinct Genres", str(len(counts)))
    with cols[2]:
        render_metric_card("Most Predicted", most_common)


# ── Model information ─────────────────────────────────────────────────────────


def render_model_info() -> None:
    """Display model architecture, dataset, and evaluation metrics."""
    st.markdown(
        """
        <div class="glass-card-accent">
        <h3 style="margin-top:0 !important">Classifiers</h3>
        <div class="info-grid" style="grid-template-columns: repeat(auto-fit, minmax(180px, 1fr))">
            <div class="info-item">
                <div class="info-item-label">Classifiers</div>
                <div class="info-item-value">RBF-SVM and PyTorch MLP</div>
            </div>
            <div class="info-item">
                <div class="info-item-label">Pipeline</div>
                <div class="info-item-value">StandardScaler → SVC</div>
            </div>
            <div class="info-item">
                <div class="info-item-label">Features</div>
                <div class="info-item-value">37 of 74</div>
            </div>
            <div class="info-item">
                <div class="info-item-label">Selection</div>
                <div class="info-item-value">Mutual Information</div>
            </div>
            <div class="info-item">
                <div class="info-item-label">Dataset</div>
                <div class="info-item-value">FMA Medium</div>
            </div>
            <div class="info-item">
                <div class="info-item-label">Genres</div>
                <div class="info-item-value">16 classes</div>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Both use 37 Mutual Information-selected features from the 74-feature schema. Each model applies its own saved training-fitted scaler.")

    # Evaluation metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="eval-card">
                <h4>Phase 5 — Mutual Information + PyTorch MLP</h4>
                <div class="eval-metric">
                    <span class="eval-metric-label">Validation macro-F1</span>
                    <span class="eval-metric-value">0.3313</span>
                </div>
                <div class="eval-metric">
                    <span class="eval-metric-label">Test macro-F1</span>
                    <span class="eval-metric-value">0.3280</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="eval-card">
                <h4>Phase 5B — Mutual Information + SVC (Deployed)</h4>
                <div class="eval-metric">
                    <span class="eval-metric-label">Validation macro-F1</span>
                    <span class="eval-metric-value">0.3828</span>
                </div>
                <div class="eval-metric">
                    <span class="eval-metric-label">Exploratory test macro-F1</span>
                    <span class="eval-metric-value">0.3150</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div class="eval-warning-note">
            <div><strong>Evaluation note:</strong> The Phase 5B test score is exploratory because the Phase 5 test partition had already been examined before this experiment was designed. It should not be interpreted as an unbiased final performance estimate.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── About / limitations ──────────────────────────────────────────────────────


def render_about() -> None:
    st.markdown(
        """
        <div class="glass-card">
        <h3 style="margin-top:0 !important">About This Project</h3>
        <p style="color:#94a3b8;line-height:1.7;font-size:0.92rem">
        This application is an academic <strong>FML (Foundations of Machine Learning)
        mini-project</strong> that explores automatic music genre classification.
        It uses the <strong>Free Music Archive (FMA) Medium</strong> dataset —
        a curated collection of 17,000 tracks spanning 16 genres.
        </p>
        <p style="color:#94a3b8;line-height:1.7;font-size:0.92rem">
        <strong>Acoustic features</strong> are extracted from raw audio using
        librosa: 20 MFCCs, 12 chroma bins, zero-crossing rate, spectral centroid,
        bandwidth, roll-off, and RMS energy — each summarised as mean and standard
        deviation, yielding <strong>74 features per track</strong>.
        </p>
        <p style="color:#94a3b8;line-height:1.7;font-size:0.92rem">
        <strong>Mutual Information feature selection</strong> identifies the 37 most
        informative features, reducing dimensionality while preserving discriminative
        power.  The selected features are fed to a <strong>Radial Basis Function (RBF)
        Support Vector Machine</strong> with train-fitted standardisation.
        </p>
        <p style="color:#94a3b8;line-height:1.7;font-size:0.92rem">
        The <strong>30-second segment workflow</strong> lets you analyse different
        portions of a song.  Genre predictions may vary between segments because
        musical style can change within a single track.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_limitations() -> None:
    st.markdown(
        """
        <div class="glass-card">
        <h3 style="margin-top:0 !important">Limitations</h3>
        <ul style="color:#94a3b8;line-height:1.8;font-size:0.9rem;padding-left:20px">
            <li>The model was trained on FMA Medium. Music outside this distribution
                (different recording quality, genres not in the 16 classes, or
                non-Western music) may produce unreliable predictions.</li>
            <li>Performance varies between genres — some genres are harder to
                distinguish than others.</li>
            <li>No confidence score is available: the SVM was trained with
                <code>probability=False</code>, so only the predicted label is
                reported. There is <strong>no confidence percentage</strong>.</li>
            <li>A single 30-second segment may not represent an entire song,
                especially for tracks with varied instrumentation.</li>
            <li>Predictions reflect statistical patterns in audio features, not
                human-level understanding of musical genre.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_genre_list(labels: list[str]) -> None:
    """Show the 16 genre labels in a compact grid."""
    items = "".join(
        f'<span style="display:inline-block;background:rgba(124,58,237,0.10);'
        f"border:1px solid rgba(124,58,237,0.18);border-radius:8px;"
        f'padding:5px 14px;margin:3px;font-size:0.82rem;color:#c084fc;'
        f'font-weight:500">{g}</span>'
        for g in labels
    )
    st.markdown(
        f'<div style="margin:10px 0">{items}</div>', unsafe_allow_html=True
    )


# ── Footer ────────────────────────────────────────────────────────────────────


def render_footer() -> None:
    st.markdown(
        """
        <div class="site-footer">
            FML Mini-Project &nbsp;·&nbsp; Music Genre Classification
            &nbsp;·&nbsp; Built with Streamlit &amp; scikit-learn
        </div>
        """,
        unsafe_allow_html=True,
    )
