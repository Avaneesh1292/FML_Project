"""Dark music-studio styling for the Streamlit music-analysis interface."""
from __future__ import annotations

import streamlit as st

CUSTOM_CSS = """
<style>
:root { color-scheme: dark; }
html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp { background: #141414; color: #F0F0F0; }
.block-container { max-width: 1160px; padding: 3rem 1.5rem 2.5rem; }
.app-heading { padding: .65rem 0 1.25rem; border-bottom: 1px solid #383838; margin-bottom: 1.35rem; }
.app-topline { display:flex; align-items:center; gap:.65rem; color:#B7C0B5; font-size:.76rem; letter-spacing:.12em; font-weight:650; text-transform:uppercase; }
.app-mark { display:grid; place-items:center; width:2rem; height:2rem; border:1px solid #484C46; border-radius:8px; color:#D6DAD2; font-size:1.15rem; }
.app-heading h1 { margin: .7rem 0 .2rem; font-size: clamp(1.8rem, 3vw, 2.35rem); line-height: 1.15; color: #F0F0F0; font-weight: 650; }
.app-intro { margin:.25rem 0 0; color:#A8A8A8; font-size:.96rem; }
.section-header { margin: 1.2rem 0 .7rem; padding-bottom: .45rem; border-bottom: 1px solid #383838; }
.section-header h2 { color: #F0F0F0; font-size: 1.08rem !important; font-weight: 620 !important; margin: 0 !important; }
.glass-card, .glass-card-accent, .prediction-card, .eval-card { background: #202020; border: 1px solid #383838; border-radius: 12px; padding: 1rem 1.1rem; }
.info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: .6rem; }
.info-item { background: #2B2B2B; border: 1px solid #3A3A3A; border-radius: 9px; padding: .65rem .75rem; }
.metric-card { background: #202020; border: 1px solid #383838; border-radius: 9px; padding: .7rem .8rem; }
.info-item-label, .metric-label { color: #A8A8A8 !important; font-size: .72rem; letter-spacing: .025em; }
.info-item-value { color: #F0F0F0 !important; font-size: .94rem; }
.segment-indicator { display:flex; justify-content:center; align-items:center; gap:.7rem; padding:.5rem; color:#D2D2D2; font-variant-numeric:tabular-nums; }
.segment-time { font-size:.98rem; font-weight:600; }
.segment-arrow { color:#A8A8A8; }
.stButton > button { border-radius:9px; min-height:2.65rem; font-weight:650; background:#B7C0B5; border:1px solid #C8CEC5; color:#171817; }
.stButton > button:hover { background:#D0D5CB; border-color:#E0E3DD; color:#141414; }
.stButton > button:focus-visible, input:focus-visible, button:focus-visible { outline:2px solid #D0D5CB !important; outline-offset:2px; }
[data-testid="stFileUploaderDropzone"] { background:#202020; border:1px dashed #555B53; border-radius:10px; }
[data-testid="stExpander"] { border:1px solid #383838; border-radius:10px; background:#1A1A1A; }
[data-testid="stSlider"] [role="slider"] { background:#B7C0B5; }
.genre-label { color:#F0F0F0; font-size:1.7rem; font-weight:680; margin:.2rem 0; overflow-wrap:anywhere; }
.prediction-meta, .prediction-model { color:#A8A8A8; font-size:.84rem; margin-top:.22rem; overflow-wrap:anywhere; }
.probability-heading { color:#B7C0B5; font-size:.78rem; margin:.75rem 0 .35rem; }
.eval-card h4 { color:#F0F0F0; font-size:.92rem; margin:.2rem 0 .55rem; }
.eval-metric { display:flex; justify-content:space-between; gap:1rem; padding:.38rem 0; border-bottom:1px solid #383838; }
.eval-metric-label { color:#A8A8A8; } .eval-metric-value { color:#F0F0F0; font-variant-numeric:tabular-nums; }
.eval-warning-note { margin-top:.7rem; padding:.65rem .8rem; border-left:3px solid #B7C0B5; background:#292B28; color:#D6D8D3; border-radius:5px; font-size:.84rem; }
.comparison-table { width:100%; border-collapse:collapse; font-size:.88rem; }
.comparison-table th, .comparison-table td { text-align:left; padding:.6rem .65rem; border-bottom:1px solid #383838; }
.comparison-table th { color:#B7C0B5; font-size:.73rem; font-weight:650; }
.model-agreement { color:#E5E8E2; background:#242624; border-left:3px solid #B7C0B5; padding:.65rem .8rem; border-radius:6px; font-size:.9rem; margin:.5rem 0 .8rem; }
.st-key-model-comparison-cards .st-key-prediction-card-svm,
.st-key-model-comparison-cards .st-key-prediction-card-mlp { min-height:31.5rem; box-sizing:border-box; }
.empty-state { display:flex; align-items:center; gap:.8rem; padding:1.1rem; border:1px dashed #484848; border-radius:10px; background:#1A1A1A; color:#A8A8A8; }
.site-footer { text-align:center; color:#888; border-top:1px solid #383838; margin-top:1.8rem; padding:1rem 0; font-size:.78rem; }
@media (max-width:640px) {
  .block-container { padding: 2.5rem .8rem 2rem; }
  .app-heading { padding-top:.35rem; margin-bottom:1rem; }
  .info-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .stRadio [role="radiogroup"] { gap:.35rem; }
  .comparison-table { font-size:.78rem; }
  .comparison-table th,.comparison-table td { padding:.45rem .35rem; }
}
</style>
"""


def inject_custom_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
