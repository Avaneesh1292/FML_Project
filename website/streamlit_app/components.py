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
        <header class="app-heading">
            <div class="app-topline"><span class="app-mark">♫</span><span>FML Mini Project&nbsp; · &nbsp;FMA Medium</span></div>
            <h1>Music Genre Classification</h1>
        </header>
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
            <div style="color:#E0E3DD;font-weight:650;font-size:1rem;margin-bottom:8px">
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
            line=dict(color="rgba(112, 132, 104, 0.68)", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(138, 155, 131, 0.16)",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Highlighted 30-second selection
    fig.add_vrect(
        x0=start_sec,
        x1=end_sec,
        fillcolor="rgba(138, 155, 131, 0.16)",
        line=dict(color="rgba(112, 132, 104, 0.74)", width=1.5, dash="dot"),
        layer="above",
        annotation_text="Selected segment",
        annotation_position="top left",
        annotation=dict(font_size=11, font_color="#D6DAD2"),
    )

    # Start / end markers
    for t, label in [(start_sec, "Start"), (end_sec, "End")]:
        fig.add_vline(
            x=t,
            line=dict(color="#B7C0B5", width=1.3),
        )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=40),
        height=180,
        xaxis=dict(
            title=dict(text="Time (seconds)", font=dict(color="#A8A8A8", size=11)),
            tickfont=dict(color="#A8A8A8", size=10),
            gridcolor="rgba(255,255,255,0.10)",
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
    model_name = result.get("model_name", "RBF-SVM")
    card_key = "prediction-card-svm" if model_name == "RBF-SVM" else "prediction-card-mlp"
    with st.container(border=True, key=card_key):
        st.caption(f"{model_name} · {result['filename']} · {format_duration(start)}–{format_duration(end)}")
        st.markdown(f"<div class='genre-label'>{escape(str(result['genre']))}</div>", unsafe_allow_html=True)
        st.caption("Predicted genre")
        decision_scores = result.get("decision_scores")
        if decision_scores:
            st.markdown("**SVM Decision Scores**")
            st.plotly_chart(
                create_decision_score_figure(decision_scores),
                width="stretch",
                config={"displayModeBar": False},
            )
            st.caption(
                "Decision scores indicate how the SVM separates classes; they are not probabilities or calibrated confidence values."
            )
        probabilities = result.get("probabilities")
        if probabilities:
            st.markdown("<div class='probability-heading'>MLP softmax outputs · not calibrated confidence</div>", unsafe_allow_html=True)
            for label, value in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:5]:
                name_col, bar_col, score_col = st.columns([1.35, 3, .55], vertical_alignment="center")
                with name_col:
                    st.caption(str(label))
                with bar_col:
                    st.progress(float(np.clip(value, 0.0, 1.0)))
                with score_col:
                    st.caption(f"{value:.0%}")


def create_decision_score_figure(scores: dict[str, float]) -> go.Figure:
    """Build a ranked horizontal chart that preserves negative SVM scores."""
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
    labels = [label for label, _ in ranked]
    values = [float(score) for _, score in ranked]
    spread = max(values) - min(values) if values else 0.0
    padding = max(spread * 0.20, 0.2)

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color="#B7C0B5",
            text=[f"{score:.2f}" for score in values],
            textfont=dict(color="#F0F0F0", size=12),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=8, r=64, t=8, b=32),
        height=max(190, len(ranked) * 38 + 50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#A8A8A8", size=11),
        hoverlabel=dict(bgcolor="#2B2B2B", bordercolor="#383838", font=dict(color="#F0F0F0", size=12)),
        xaxis=dict(
            title="Decision score",
            range=[min(0.0, min(values, default=0.0) - padding), max(0.0, max(values, default=0.0) + padding)],
            zeroline=True,
            zerolinecolor="#A8A8A8",
            zerolinewidth=1,
            gridcolor="rgba(255,255,255,0.10)",
            tickfont=dict(color="#A8A8A8", size=10),
            tickformat=".2f",
            title_font=dict(color="#A8A8A8", size=10),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#F0F0F0", size=10),
        ),
        showlegend=False,
    )
    return fig


def render_model_comparison(results: list[dict[str, Any]]) -> None:
    """Show existing same-segment predictions for both selected models."""
    by_model = {item.get("model_name", "RBF-SVM"): item for item in results}
    svm = by_model.get("RBF-SVM")
    mlp = by_model.get("PyTorch MLP")
    if svm is None or mlp is None:
        return
    if svm["genre"] == mlp["genre"]:
        st.markdown(f"<div class='model-agreement'>Both models predicted <strong>{escape(str(svm['genre']))}</strong> for this segment.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='model-agreement'>Predictions differ: RBF-SVM — <strong>{escape(str(svm['genre']))}</strong>; PyTorch MLP — <strong>{escape(str(mlp['genre']))}</strong>.</div>", unsafe_allow_html=True)
    with st.container(key="model-comparison-cards"):
        left, right = st.columns(2, gap="medium")
        with left:
            render_prediction_card(svm)
        with right:
            render_prediction_card(mlp)


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
            f"<td style='color:#E0E3DD;font-weight:600'>{escape(str(r['genre']))}</td>"
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
                colorscale=[[0, "#697363"], [0.5, "#929C8A"], [1, "#B7C0B5"]],
                line=dict(width=0),
            ),
            text=values,
            textposition="outside",
            textfont=dict(color="#E0E3DD", size=12, family="Inter"),
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=40),
        height=320,
        xaxis=dict(
            tickfont=dict(color="#D0D0D0", size=11),
            gridcolor="rgba(255,255,255,0.10)",
        ),
        yaxis=dict(
            tickfont=dict(color="#A8A8A8", size=10),
            gridcolor="rgba(255,255,255,0.10)",
            title=dict(text="Segments", font=dict(color="#A8A8A8", size=11)),
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
        "#929C8A", "#A8B0A2", "#B7C0B5", "#929C8A", "#A8B0A2",
        "#A2AA9A", "#929C8A", "#B7C0B5", "#B7C0B5", "#B1B9AC",
        "#929C8A", "#D0D5CB", "#A8B0A2", "#B7C0B5", "#929C8A", "#A2AA9A",
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
            textfont=dict(color="#171817", size=11, family="Inter"),
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
            title=dict(text="Time (seconds)", font=dict(color="#A8A8A8", size=11)),
            tickfont=dict(color="#A8A8A8", size=10),
            gridcolor="rgba(255,255,255,0.10)",
        ),
        yaxis=dict(
            tickfont=dict(color="#D0D0D0", size=11),
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
                <div class="info-item-label">Preprocessing</div>
                <div class="info-item-value">Saved StandardScaler per model</div>
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
        <p style="color:#A8A8A8;line-height:1.7;font-size:0.92rem">
        This application is an academic <strong>FML (Foundations of Machine Learning)
        mini-project</strong> that explores automatic music genre classification.
        It uses the <strong>Free Music Archive (FMA) Medium</strong> dataset —
        a curated collection of 17,000 tracks spanning 16 genres.
        </p>
        <p style="color:#A8A8A8;line-height:1.7;font-size:0.92rem">
        <strong>Acoustic features</strong> are extracted from raw audio using
        librosa: 20 MFCCs, 12 chroma bins, zero-crossing rate, spectral centroid,
        bandwidth, roll-off, and RMS energy — each summarised as mean and standard
        deviation, yielding <strong>74 features per track</strong>.
        </p>
        <p style="color:#A8A8A8;line-height:1.7;font-size:0.92rem">
        <strong>Mutual Information feature selection</strong> identifies the 37 most
        informative features, reducing dimensionality while preserving discriminative
        power.  The selected features are used by both classifiers with their own
        saved, training-fitted standardisation.
        </p>
        <p style="color:#A8A8A8;line-height:1.7;font-size:0.92rem">
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
        <ul style="color:#A8A8A8;line-height:1.8;font-size:0.9rem;padding-left:20px">
            <li>The model was trained on FMA Medium. Music outside this distribution
                (different recording quality, genres not in the 16 classes, or
                non-Western music) may produce unreliable predictions.</li>
            <li>Performance varies between genres — some genres are harder to
                distinguish than others.</li>
            <li>The SVM provides no probability output. The MLP's displayed
                softmax outputs are not calibrated confidence estimates.</li>
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
        f'<span style="display:inline-block;background:#2B2B2B;'
        f"border:1px solid #484848;border-radius:8px;"
        f'padding:5px 12px;margin:3px;font-size:0.82rem;color:#E0E3DD;'
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
            &nbsp;·&nbsp; Built with Streamlit, scikit-learn &amp; PyTorch
        </div>
        """,
        unsafe_allow_html=True,
    )
