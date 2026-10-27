"""FML Music Genre Classification — Interactive Web Application.

Launch with:  python -m streamlit run website/app.py
"""
from __future__ import annotations

import sys
import hashlib
from datetime import datetime
from pathlib import Path

# ── Project root setup ────────────────────────────────────────────────────────
# Add the *project root* (parent of website/) so ``from src.prediction import …``
# works, and add *this directory* so ``from web_utils import …`` works.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIS_DIR))

import numpy as np
import streamlit as st

from src.prediction import load_predictor, predict_features

from streamlit_app.components import (
    create_genre_distribution_chart,
    create_prediction_timeline,
    create_waveform_figure,
    render_about,
    render_analysis_summary,
    render_audio_info,
    render_comparison_table,
    render_footer,
    render_genre_list,
    render_page_header,
    render_limitations,
    render_metric_card,
    render_model_info,
    render_prediction_card,
    render_section_header,
    render_segment_indicator,
)
from streamlit_app.styles import inject_custom_css
from web_utils import (
    extract_segment,
    format_duration,
    generate_waveform_data,
    load_audio_from_upload,
    extract_segment_features,
    segment_to_wav_bytes,
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Genre Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = _PROJECT_ROOT / "models" / "phase5b_svm" / "phase5b_svm__mutual_information.joblib"
MLP_PATH = _THIS_DIR / "models" / "phase5_mutual_information_mlp_cpu.pt"
SEGMENT_DURATION = 30.0  # seconds — fixed
MAX_HISTORY = 50  # bounded session history

# ── Model loading (cached) ────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _load_model_bundle():
    """Load and validate the prediction bundle once, cached across reruns."""
    return load_predictor(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def _load_mlp_bundle():
    from mlp_inference import load_mlp_predictor
    return load_mlp_predictor(MLP_PATH)


try:
    _bundle = _load_model_bundle()
    _model_loaded = True
    _model_error = ""
    _genre_labels: list[str] = list(_bundle["labels"])
except Exception as exc:
    _bundle = None
    _model_loaded = False
    _model_error = str(exc)
    _genre_labels = []


# ── Session state initialisation ──────────────────────────────────────────────


def _init_state() -> None:
    defaults = {
        "audio_waveform": None,
        "audio_sr": 22050,
        "audio_duration": 0.0,
        "audio_filename": "",
        "audio_filesize": 0,
        "audio_channels": 0,
        "audio_original_sr": 0,
        "uploaded_file_id": None,
        "feature_cache_key": None,
        "feature_cache": None,
        "selected_model": "RBF-SVM",
        "segment_start": 0.0,
        "current_results": [],
        "analysis_counter": 0,
        "session_history": [],
        "last_prediction": None,
        "upload_key_counter": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════

render_page_header()

if not _model_loaded:
    st.error(f"⚠️ Could not load the prediction model. {_model_error}")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  UPLOAD SECTION
# ══════════════════════════════════════════════════════════════════════════════

render_section_header("", "Upload Audio")

upload_key = f"uploader_{st.session_state.upload_key_counter}"
uploaded_file = st.file_uploader(
    "Drag and drop or browse an audio file (MP3, WAV)",
    type=["mp3", "wav", "ogg", "flac"],
    key=upload_key,
    help="Maximum 50 MB. The audio is processed locally — nothing is sent to external servers.",
)

# Handle upload / clear
if uploaded_file is not None:
    file_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

    if file_id != st.session_state.uploaded_file_id:
        # ── New file: decode and reset analysis state ────────────────
        with st.spinner("Decoding audio…"):
            try:
                info = load_audio_from_upload(
                    uploaded_file.getvalue(), uploaded_file.name
                )
                st.session_state.audio_waveform = info["waveform"]
                st.session_state.audio_sr = info["sr"]
                st.session_state.audio_duration = info["duration"]
                st.session_state.audio_filename = uploaded_file.name
                st.session_state.audio_filesize = uploaded_file.size
                st.session_state.audio_channels = info["channels"]
                st.session_state.audio_original_sr = info["original_sr"]
                st.session_state.uploaded_file_id = file_id
                st.session_state.segment_start = 0.0
                st.session_state.current_results = []
                st.session_state.analysis_counter = 0
                st.session_state.last_prediction = None
                st.session_state.feature_cache_key = None
                st.session_state.feature_cache = None
            except Exception as exc:
                st.error(f"Could not decode this audio file. {exc}")
                st.session_state.audio_waveform = None
                st.session_state.uploaded_file_id = None
                st.session_state.feature_cache_key = None
                st.session_state.feature_cache = None
else:
    # File uploader is empty — clear audio state if something was loaded
    if st.session_state.audio_waveform is not None:
        # Save any results to session history before clearing
        if st.session_state.current_results:
            st.session_state.session_history = (
                st.session_state.current_results
                + st.session_state.session_history
            )[:MAX_HISTORY]
        st.session_state.audio_waveform = None
        st.session_state.uploaded_file_id = None
        st.session_state.current_results = []
        st.session_state.last_prediction = None
        st.session_state.analysis_counter = 0
        st.session_state.feature_cache_key = None
        st.session_state.feature_cache = None


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO LOADED — main analysis flow
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.audio_waveform is not None:
    waveform: np.ndarray = st.session_state.audio_waveform
    sr: int = st.session_state.audio_sr
    duration: float = st.session_state.audio_duration
    filename: str = st.session_state.audio_filename

    # ── Audio info card ──────────────────────────────────────────────
    render_audio_info(
        filename=filename,
        filesize=st.session_state.audio_filesize,
        duration=duration,
        sr=sr,
        channels=st.session_state.audio_channels,
        original_sr=st.session_state.audio_original_sr,
    )

    # ── Full-track audio player ──────────────────────────────────────
    with st.expander("Play full track", expanded=False):
        full_wav = segment_to_wav_bytes(waveform, sr)
        st.audio(full_wav, format="audio/wav")

    # ── Track too short? ─────────────────────────────────────────────
    if duration < SEGMENT_DURATION:
        st.warning(
            f"This track is only **{format_duration(duration)}** long. "
            f"A minimum of **{format_duration(SEGMENT_DURATION)}** is required "
            f"for genre prediction. Please upload a longer track."
        )
    else:
        # ══════════════════════════════════════════════════════════════
        #  WAVEFORM & SEGMENT SELECTION
        # ══════════════════════════════════════════════════════════════

        render_section_header("", "Select a 30-Second Segment")

        max_start = duration - SEGMENT_DURATION

        # Waveform data (cached per file)
        @st.cache_data(show_spinner=False)
        def _waveform_data(_wf_hash: str, _wf: np.ndarray, _sr: int):
            return generate_waveform_data(_wf, _sr)

        wf_hash = f"{filename}_{len(waveform)}_{sr}"
        times, amplitudes = _waveform_data(wf_hash, waveform, sr)

        # Start-time selection
        if max_start < 0.5:
            st.info(
                f"Track duration is {format_duration(duration)}. "
                f"Analysing the 30-second window starting at 00:00."
            )
            start_sec = 0.0
        else:
            start_sec = float(st.slider(
                "Segment start time",
                min_value=0.0,
                max_value=float(round(max_start, 1)),
                value=float(min(st.session_state.segment_start, max_start)),
                step=0.5,
                format="%.1f s",
                help="Drag to choose where the 30-second analysis window begins.",
            ))
        st.session_state.segment_start = start_sec
        end_sec = start_sec + SEGMENT_DURATION

        # Segment indicator
        render_segment_indicator(start_sec, end_sec)

        # Plotly waveform
        wf_fig = create_waveform_figure(times, amplitudes, start_sec, end_sec, duration)
        st.plotly_chart(wf_fig, use_container_width=True, config={"displayModeBar": False})

        # Extract the segment for preview and (potentially) prediction
        try:
            segment = extract_segment(waveform, sr, start_sec, SEGMENT_DURATION)
        except Exception as exc:
            st.error(f"Segment extraction error: {exc}")
            segment = None

        if segment is not None:
            # Segment preview player
            segment_wav = segment_to_wav_bytes(segment, sr)
            st.markdown(
                "<p style='color:#64748b;font-size:0.82rem;margin-bottom:4px'>"
                "▶ Preview selected segment</p>",
                unsafe_allow_html=True,
            )
            st.audio(segment_wav, format="audio/wav")

            # Precise numeric input (fallback)
            if max_start >= 0.5:
                with st.expander("Precise start-time input", expanded=False):
                    precise_start = float(st.number_input(
                        "Start time (seconds)",
                        min_value=0.0,
                        max_value=float(round(max_start, 2)),
                        value=float(round(start_sec, 2)),
                        step=1.0,
                        format="%.2f",
                    ))
                    if abs(precise_start - start_sec) > 0.01:
                        st.session_state.segment_start = precise_start
                        st.rerun()

            # ══════════════════════════════════════════════════════════
            #  ANALYZE BUTTON
            # ══════════════════════════════════════════════════════════
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                selected_model = st.radio(
                    "Classification model",
                    options=["RBF-SVM", "PyTorch MLP"],
                    horizontal=True,
                    key="selected_model",
                    help="Two classifiers with their own saved preprocessing.",
                )
                analyze_clicked = st.button(
                    "Classify Genre",
                    use_container_width=True,
                    type="primary",
                )

            if analyze_clicked:
                with st.spinner("Preparing features and predicting genre…"):
                    try:
                        active_bundle = _bundle if selected_model == "RBF-SVM" else _load_mlp_bundle()
                        cache_key = f"{st.session_state.uploaded_file_id}:{start_sec:.3f}:{SEGMENT_DURATION:.1f}"
                        if st.session_state.feature_cache_key != cache_key:
                            st.session_state.feature_cache = extract_segment_features(
                                segment, sr, active_bundle["audio_config"]
                            )
                            st.session_state.feature_cache_key = cache_key
                        result = predict_features(st.session_state.feature_cache, active_bundle)
                        predicted_genre = result["genre"]

                        new_entry = {
                            "segment_number": st.session_state.analysis_counter + 1,
                            "filename": filename,
                            "start": start_sec,
                            "end": end_sec,
                            "genre": predicted_genre,
                            "model_name": selected_model,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        }
                        st.session_state.analysis_counter += 1
                        st.session_state.current_results.append(new_entry)
                        st.session_state.last_prediction = new_entry
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

        # ══════════════════════════════════════════════════════════════
        #  PREDICTION RESULTS
        # ══════════════════════════════════════════════════════════════

        if st.session_state.last_prediction:
            render_section_header("", "Prediction Result")
            render_prediction_card(st.session_state.last_prediction)

            # Segment audio for the latest result
            try:
                last = st.session_state.last_prediction
                res_segment = extract_segment(
                    waveform, sr, last["start"], SEGMENT_DURATION
                )
                res_wav = segment_to_wav_bytes(res_segment, sr)
                with st.expander("Listen to analyzed segment", expanded=False):
                    st.audio(res_wav, format="audio/wav")
            except Exception:
                pass  # non-critical

        # ══════════════════════════════════════════════════════════════
        #  MULTI-SEGMENT COMPARISON
        # ══════════════════════════════════════════════════════════════

        results = st.session_state.current_results
        if results:
            render_section_header("", "Segment Comparison")

            render_comparison_table(results)

            # Remove individual entry
            if len(results) > 1:
                col_rm, col_clr, _ = st.columns([1, 1, 2])
                with col_rm:
                    seg_nums = [r["segment_number"] for r in results]
                    remove_num = st.selectbox(
                        "Remove segment",
                        seg_nums,
                        index=len(seg_nums) - 1,
                        label_visibility="collapsed",
                    )
                    if st.button("Remove selected", key="btn_remove"):
                        st.session_state.current_results = [
                            r for r in results if r["segment_number"] != remove_num
                        ]
                        if (
                            st.session_state.last_prediction
                            and st.session_state.last_prediction["segment_number"]
                            == remove_num
                        ):
                            st.session_state.last_prediction = (
                                st.session_state.current_results[-1]
                                if st.session_state.current_results
                                else None
                            )
                        st.rerun()
                with col_clr:
                    if st.button("Clear all", key="btn_clear_all"):
                        st.session_state.current_results = []
                        st.session_state.last_prediction = None
                        st.session_state.analysis_counter = 0
                        st.rerun()
            elif len(results) == 1:
                if st.button("Clear result", key="btn_clear_single"):
                    st.session_state.current_results = []
                    st.session_state.last_prediction = None
                    st.session_state.analysis_counter = 0
                    st.rerun()

            # ══════════════════════════════════════════════════════════
            #  VISUAL ANALYTICS
            # ══════════════════════════════════════════════════════════

            if len(results) >= 2:
                render_section_header("", "Visual Analytics")

                st.caption(
                    "These charts show predictions for the segments you analysed "
                    "— they are **not** ground-truth labels or model accuracy metrics."
                )

                render_analysis_summary(results)
                st.markdown("<br>", unsafe_allow_html=True)

                tab_dist, tab_timeline = st.tabs(
                    ["Genre Distribution", "Prediction Timeline"]
                )
                with tab_dist:
                    fig_dist = create_genre_distribution_chart(results)
                    st.plotly_chart(
                        fig_dist,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                with tab_timeline:
                    fig_tl = create_prediction_timeline(results)
                    st.plotly_chart(
                        fig_tl,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION HISTORY (across songs)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.session_history:
    with st.expander("Recent session history", expanded=False):
        st.caption("Previous analyses from earlier uploads in this session.")
        render_comparison_table(st.session_state.session_history[:20])
        if st.button("Clear session history", key="btn_clear_hist"):
            st.session_state.session_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL INFORMATION & ABOUT
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("Model details and evaluation", expanded=False):
    render_model_info()

if _genre_labels:
    st.markdown(
        "<p style='color:#94a3b8;font-size:0.9rem;margin-top:12px'>"
        "<strong>Supported genres (16 classes):</strong></p>",
        unsafe_allow_html=True,
    )
    render_genre_list(_genre_labels)

with st.expander("About this project", expanded=False):
    render_about()

with st.expander("Limitations", expanded=False):
    render_limitations()

render_footer()
