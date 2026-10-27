"""Restrained charcoal theme for the Streamlit interface."""
from __future__ import annotations

import streamlit as st

CUSTOM_CSS = """
<style>
:root { color-scheme: dark; }
html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp { background: radial-gradient(ellipse at 50% -20%, #20212a 0%, #111216 42%, #0c0d10 100%); color: #e9e9ed; }
.block-container { max-width: 1040px; padding-top: 2rem; padding-bottom: 2.5rem; }
.app-heading { padding: 1.5rem 0 1.1rem; border-bottom: 1px solid #2b2c32; margin-bottom: 1.7rem; }
.app-kicker { color: #aaa7bd; font-size: .72rem; letter-spacing: .13em; font-weight: 650; }
.app-heading h1 { margin: .45rem 0 .35rem; font-size: clamp(1.8rem, 4vw, 2.5rem); line-height: 1.14; color: #f2f1f5; font-weight: 650; }
.app-heading p { color: #aaaab3; margin: 0; font-size: .98rem; }
.section-header { margin: 1.5rem 0 .75rem; padding-bottom: .55rem; border-bottom: 1px solid #292a30; }
.section-header h2 { color: #e8e7ed; font-size: 1.1rem !important; font-weight: 600 !important; margin: 0 !important; }
.glass-card, .glass-card-accent, .prediction-card, .eval-card { background: rgba(27, 28, 34, .82); border: 1px solid #34353d; border-radius: 13px; padding: 1rem 1.15rem; }
.info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(125px, 1fr)); gap: .6rem; }
.info-item { background: #202127; border: 1px solid #34353d; border-radius: 9px; padding: .7rem; }
.info-item-label, .metric-label { color: #a4a3ad !important; font-size: .7rem; letter-spacing: .06em; text-transform: uppercase; }
.info-item-value { color: #e7e6eb !important; font-size: .95rem; }
.segment-indicator { display:flex; justify-content:center; align-items:center; gap: .8rem; padding:.65rem; color:#d2cce8; font-variant-numeric: tabular-nums; }
.segment-time { font-size: 1rem; font-weight: 600; }
.segment-arrow { color:#888793; }
.stButton > button { border-radius: 9px; min-height: 2.7rem; font-weight: 600; background:#65558f; border:1px solid #8574b2; color:#fff; }
.stButton > button:hover { background:#7764a7; border-color:#a293ca; }
.stButton > button:focus-visible, input:focus-visible, button:focus-visible { outline: 2px solid #b9a7ef !important; outline-offset: 2px; }
[data-testid="stFileUploaderDropzone"] { background:#1b1c22; border:1px dashed #54535f; border-radius:11px; }
[data-testid="stExpander"] { border:1px solid #34353d; border-radius:10px; background:rgba(27,28,34,.72); }
[data-testid="stSlider"] [role="slider"] { background:#a99ad2; }
.genre-label { color:#d6c9f4; font-size:clamp(1.6rem, 4vw, 2.25rem); font-weight:650; margin:.25rem 0; }
.prediction-meta, .prediction-model { color:#aaaab3; font-size:.86rem; margin-top:.25rem; overflow-wrap:anywhere; }
.eval-card h4 { color:#d7cdee; font-size:.95rem; }
.eval-metric { display:flex; justify-content:space-between; gap:1rem; padding:.4rem 0; border-bottom:1px solid #34353d; }
.eval-metric-label { color:#aaaab3; } .eval-metric-value { color:#eae8ef; font-variant-numeric:tabular-nums; }
.eval-warning-note { margin-top:.8rem; padding:.7rem .85rem; border-left:3px solid #9483bd; background:#24232b; color:#c3c1ca; border-radius:6px; font-size:.86rem; }
.comparison-table { width:100%; border-collapse:collapse; font-size:.88rem; }
.comparison-table th, .comparison-table td { text-align:left; padding:.6rem .7rem; border-bottom:1px solid #34353d; }
.comparison-table th { color:#c9c1dc; font-size:.74rem; text-transform:uppercase; letter-spacing:.05em; }
.site-footer { text-align:center; color:#84838d; border-top:1px solid #2c2d33; margin-top:2.2rem; padding:1.1rem 0; font-size:.78rem; }
@media (max-width: 640px) { .block-container { padding:1rem .85rem 2rem; } .app-heading { padding-top:.7rem; } .info-grid { grid-template-columns:repeat(2, minmax(0,1fr)); } .stRadio [role="radiogroup"] { gap:.35rem; } .comparison-table { font-size:.78rem; } .comparison-table th, .comparison-table td { padding:.48rem .35rem; } }
</style>
"""


def inject_custom_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
