"""Custom CSS theme for the FML Music Genre Classification website.

Injects a premium dark aesthetic with violet/indigo accents, glassmorphism
cards, and Inter typography via Google Fonts.
"""
from __future__ import annotations

import streamlit as st

GOOGLE_FONTS_LINK = (
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Inter:wght@300;400;500;600;700;800;900&display=swap" '
    'rel="stylesheet">'
)

CUSTOM_CSS = """
<style>
/* ===================================================================
   FML Music Genre Classifier — Premium Dark Theme
   =================================================================== */

/* ── Google Fonts ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global base ─────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(180deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a12 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1100px;
}

/* ── Typography ──────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6,
p, li,
.stMarkdown, .stText {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

h1, h2, h3 { color: #e2e8f0 !important; }
p           { color: #cbd5e1 !important; }

/* ── Hero section ────────────────────────────────────────────────── */
.hero-container {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    margin-bottom: 1rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(124, 58, 237, 0.13);
    border: 1px solid rgba(124, 58, 237, 0.30);
    border-radius: 20px;
    padding: 6px 18px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #a78bfa;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
}
.hero-title {
    font-size: 3.1rem;
    font-weight: 800;
    line-height: 1.12;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #c084fc 0%, #7c3aed 35%, #6366f1 65%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-size: 1.12rem;
    color: #94a3b8 !important;
    font-weight: 400;
    max-width: 620px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ── Glass cards ─────────────────────────────────────────────────── */
.glass-card {
    background: rgba(18, 18, 35, 0.55);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(124, 58, 237, 0.10);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(124, 58, 237, 0.22);
    box-shadow: 0 4px 24px rgba(124, 58, 237, 0.07);
}
.glass-card-accent {
    background: rgba(18, 18, 35, 0.55);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(124, 58, 237, 0.22);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 16px;
    box-shadow: 0 0 30px rgba(124, 58, 237, 0.05);
}

/* ── Prediction result card ──────────────────────────────────────── */
.prediction-card {
    background: linear-gradient(135deg,
        rgba(124, 58, 237, 0.10) 0%,
        rgba(99, 102, 241, 0.07) 100%);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(124, 58, 237, 0.28);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.prediction-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #7c3aed, #6366f1, #8b5cf6, #7c3aed);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
.genre-label {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c084fc, #7c3aed, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 12px 0;
}
.prediction-meta {
    color: #94a3b8 !important;
    font-size: 0.88rem;
    margin-top: 6px;
}

/* ── Metric cards ────────────────────────────────────────────────── */
.metric-card {
    background: rgba(18, 18, 35, 0.45);
    border: 1px solid rgba(124, 58, 237, 0.10);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
}
.metric-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #c084fc !important;
    margin: 4px 0;
}
.metric-label {
    font-size: 0.72rem;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}

/* ── Section headers ─────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 2.8rem 0 1.2rem;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(124, 58, 237, 0.10);
}
.section-header h2 {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    padding: 0 !important;
}
.section-icon { font-size: 1.25rem; }

/* ── Info grid ───────────────────────────────────────────────────── */
.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin: 12px 0;
}
.info-item {
    background: rgba(18, 18, 35, 0.40);
    border: 1px solid rgba(124, 58, 237, 0.08);
    border-radius: 10px;
    padding: 14px 10px;
    text-align: center;
}
.info-item-label {
    font-size: 0.68rem;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
    margin-bottom: 3px;
}
.info-item-value {
    font-size: 1rem;
    color: #e2e8f0 !important;
    font-weight: 600;
}

/* ── Segment indicator ───────────────────────────────────────────── */
.segment-indicator {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 14px;
    padding: 12px 20px;
    background: rgba(124, 58, 237, 0.07);
    border: 1px solid rgba(124, 58, 237, 0.14);
    border-radius: 12px;
    margin: 10px 0;
}
.segment-time {
    font-size: 1.1rem;
    font-weight: 600;
    color: #c084fc !important;
    font-variant-numeric: tabular-nums;
}
.segment-arrow {
    color: #475569 !important;
    font-size: 1rem;
}
.segment-duration-label {
    font-size: 0.82rem;
    color: #64748b !important;
    font-weight: 500;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.6rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 2px 8px rgba(124, 58, 237, 0.18) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124, 58, 237, 0.32) !important;
    filter: brightness(1.08) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── File uploader ───────────────────────────────────────────────── */
[data-testid="stFileUploader"] > div:first-child {
    background: rgba(18, 18, 35, 0.35) !important;
    border: 2px dashed rgba(124, 58, 237, 0.22) !important;
    border-radius: 16px !important;
    transition: border-color 0.3s ease !important;
}
[data-testid="stFileUploader"] > div:first-child:hover {
    border-color: rgba(124, 58, 237, 0.45) !important;
}

/* File uploader button styling: clean, cohesive 'Browse files' */
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    box-shadow: 0 2px 10px rgba(124, 58, 237, 0.25) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.4) !important;
    filter: brightness(1.08) !important;
}

/* Hide the raw ligature text 'upload' from the icon span */
[data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"] {
    display: none !important;
}

/* Display clean, readable 'Browse files' button label */
[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p {
    font-size: 0 !important;
    line-height: 0 !important;
    margin: 0 !important;
    visibility: hidden !important;
}
[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p::after {
    content: "Browse files" !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    color: #ffffff !important;
    visibility: visible !important;
    display: inline-block !important;
    line-height: 1.4 !important;
}

/* ── Slider ──────────────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background-color: #7c3aed !important;
}

/* ── Expander ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(18, 18, 35, 0.35);
    border: 1px solid rgba(124, 58, 237, 0.10);
    border-radius: 14px;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(18, 18, 35, 0.40);
    border-radius: 10px 10px 0 0;
    padding: 8px 20px;
    color: #94a3b8;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(124, 58, 237, 0.18) !important;
    color: #c084fc !important;
    border-bottom: 2px solid #7c3aed !important;
}

/* ── Alert / info boxes ──────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* ── Divider ─────────────────────────────────────────────────────── */
hr {
    border-color: rgba(124, 58, 237, 0.08) !important;
    margin: 2rem 0 !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.15); }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.45); }

/* ── Hide Streamlit chrome ───────────────────────────────────────── */
#MainMenu  { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: hidden; }

/* ── Footer ──────────────────────────────────────────────────────── */
.site-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #475569 !important;
    font-size: 0.78rem;
    border-top: 1px solid rgba(124, 58, 237, 0.07);
    margin-top: 3rem;
}
.site-footer a {
    color: #7c3aed;
    text-decoration: none;
}

/* ── Responsive ──────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .hero-title         { font-size: 2rem; }
    .hero-subtitle      { font-size: 0.95rem; }
    .genre-label        { font-size: 1.7rem; }
    .glass-card,
    .glass-card-accent,
    .prediction-card    { padding: 16px; border-radius: 12px; }
    .info-grid          { grid-template-columns: repeat(2, 1fr); }
    .segment-indicator  { flex-wrap: wrap; }
}

/* ── Model-info cards (for the evaluation section) ───────────────── */
.eval-card {
    background: rgba(18, 18, 35, 0.45);
    border: 1px solid rgba(124, 58, 237, 0.12);
    border-radius: 14px;
    padding: 22px;
}
.eval-card h4 {
    font-size: 1rem !important;
    color: #c084fc !important;
    font-weight: 700 !important;
    margin-bottom: 10px !important;
}
.eval-metric {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.9rem;
}
.eval-metric-label { color: #94a3b8 !important; }
.eval-metric-value { color: #e2e8f0 !important; font-weight: 600; font-variant-numeric: tabular-nums; }

/* ── Evaluation note ─────────────────────────────────────────────── */
.eval-warning-note {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: rgba(124, 58, 237, 0.05);
    border: 1px solid rgba(124, 58, 237, 0.14);
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 14px;
    color: #94a3b8 !important;
    font-size: 0.83rem;
    line-height: 1.55;
}
.eval-warning-note .note-icon {
    font-size: 0.95rem;
    flex-shrink: 0;
    margin-top: 1px;
}
.eval-warning-note strong {
    color: #c084fc !important;
}

/* ── Comparison table ────────────────────────────────────────────── */
.comparison-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    font-size: 0.88rem;
}
.comparison-table thead th {
    background: rgba(124, 58, 237, 0.12);
    color: #c084fc;
    font-weight: 600;
    padding: 12px 14px;
    text-align: left;
    border-bottom: 1px solid rgba(124, 58, 237, 0.15);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.comparison-table tbody td {
    padding: 10px 14px;
    color: #e2e8f0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    background: rgba(18, 18, 35, 0.3);
}
.comparison-table tbody tr:hover td {
    background: rgba(124, 58, 237, 0.06);
}
</style>
"""


def inject_custom_css() -> None:
    """Inject the custom CSS theme and Google Fonts into the Streamlit page."""
    st.markdown(GOOGLE_FONTS_LINK, unsafe_allow_html=True)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
