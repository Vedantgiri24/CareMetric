"""
CareMetric — Predictive Premium Intelligence
=====================================
A clinical, data-forward insurance premium estimator. Intake reads like
a light health-analytics dashboard; the result lands on a single,
well-composed metric card: a large headline figure, a radial position
ring, a risk tier badge, and supporting stat cells underneath.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import hashlib
import os

# Files created by the training script (train_model.py) / dataset export.
# Keep both of these in the same directory as this app.py when deploying.
MODEL_PATH = "insurance_model_bundle.pkl"
DATA_PATH = "insurance.csv"

# ---------------------------------------------------------------
# PAGE CONFIG (must be first Streamlit call)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="CareMetric — Predictive Premium Intelligence",
    page_icon="🩺",
    layout="centered",
)

# ---------------------------------------------------------------
# DESIGN TOKENS — CSS
# Clinical mint-white surface, deep forest-ink text, emerald primary,
# amber/coral risk accents. Light, precise, dashboard-grade.
# ---------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600;700&display=swap" rel="stylesheet">
<style>
    :root {
        --bg: #F1F7F4;
        --surface: #FFFFFF;
        --surface-alt: #EAF2ED;
        --ink: #10261F;
        --ink-dim: #5C7268;
        --line: #DCE7E1;
        --primary: #157A5B;
        --primary-dim: rgba(21,122,91,0.10);
        --primary-line: rgba(21,122,91,0.35);
        --amber: #B87A22;
        --amber-dim: rgba(184,122,34,0.10);
        --coral: #BD4438;
        --coral-dim: rgba(189,68,56,0.10);
        --shadow: 0 10px 28px rgba(16,38,31,0.08);
        --shadow-lg: 0 18px 40px rgba(16,38,31,0.12);
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }
    .stApp {
        background-color: var(--bg);
        background-image:
            linear-gradient(rgba(16,38,31,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(16,38,31,0.035) 1px, transparent 1px);
        background-size: 28px 28px;
    }

    h1, h2, h3 { font-family: 'Sora', sans-serif !important; color: var(--ink) !important; font-weight: 700 !important; letter-spacing: -0.01em; }

    /* --- Brand header --- */
    .brand-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.7rem 1.1rem;
        margin-bottom: 1.3rem;
        box-shadow: var(--shadow);
    }
    .brand-bar .brand {
        font-family: 'Sora', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: -0.01em;
        color: var(--ink);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .brand-bar .brand .mark {
        width: 10px; height: 10px; border-radius: 3px;
        background: var(--primary);
    }
    .brand-bar .status {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: var(--ink-dim);
        display: flex; align-items: center; gap: 0.4rem;
        letter-spacing: 0.02em;
    }
    .brand-bar .status .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--primary); }

    .page-heading { margin-bottom: 1.3rem; }
    .page-heading .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--primary);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 0.35rem;
    }
    .page-heading h1 { font-size: 1.7rem !important; margin: 0 0 0.35rem 0 !important; }
    .page-heading .sub { color: var(--ink-dim); font-size: 0.92rem; max-width: 34rem; line-height: 1.5; }

    /* --- Intake card --- */
    .intake-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1.2rem 1.3rem 1rem 1.3rem;
        margin-bottom: 1.2rem;
        box-shadow: var(--shadow);
    }
    .card-title {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 1.02rem;
        color: var(--ink);
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.2rem;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid var(--line);
    }
    .card-title .card-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 500;
        font-size: 0.66rem;
        color: var(--ink-dim);
        letter-spacing: 0.04em;
    }
    .section-label {
        display: flex; align-items: center; gap: 0.4rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--primary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1rem 0 0.55rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px dashed var(--primary-line);
    }
    .section-label:first-of-type { margin-top: 0.7rem; }
    .section-label svg { width: 14px; height: 14px; flex-shrink: 0; }

    .field-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--ink-dim);
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .field-tag {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem;
        font-weight: 700;
        padding: 0.18rem 0.6rem;
        border-radius: 4px;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .tag-low { color: var(--primary); background: var(--primary-dim); }
    .tag-moderate { color: var(--amber); background: var(--amber-dim); }
    .tag-high { color: var(--coral); background: var(--coral-dim); }
    .tag-dim { color: var(--ink-dim); background: var(--surface-alt); }

    /* Streamlit bordered container -> field tile */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        background: var(--surface-alt) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] { gap: 0.3rem; }

    /* --- Buttons --- */
    button[kind="primary"] {
        background: var(--primary) !important;
        color: #FFFFFF !important;
        border: 1px solid #0F5E45 !important;
        border-radius: 9px !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em;
        box-shadow: 0 4px 12px rgba(21,122,91,0.28) !important;
    }
    button[kind="primary"]:hover { background: #12694E !important; }
    button[kind="primary"] p { color: #FFFFFF !important; font-weight: 700; }

    button[kind="secondary"] {
        background-color: var(--surface) !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
    }
    button[kind="secondary"]:hover { border-color: var(--primary) !important; color: var(--primary) !important; }
    button[kind="secondary"] p { color: inherit !important; }

    .stDownloadButton>button {
        background-color: var(--surface);
        color: var(--primary) !important;
        border: 1px solid var(--primary-line);
        border-radius: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stDownloadButton>button:hover { background-color: var(--primary); color: #FFFFFF !important; border-color: #0F5E45; }
    .stDownloadButton>button p { color: inherit !important; }


    /* =============================================================
       CONTRAST / WIDGET OVERRIDES (light theme)
       ============================================================= */
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
    label, .stMarkdown, .stCaption, p, span, strong, em, b, li,
    [data-testid="stMarkdownContainer"] * { color: var(--ink) !important; }
    .page-heading .sub, .page-heading .eyebrow { color: inherit !important; }

    div[data-baseweb="select"] > div {
        background-color: var(--surface) !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] span, div[data-baseweb="select"] div { color: var(--ink) !important; }
    div[data-baseweb="select"] svg { fill: var(--ink) !important; }
    ul[data-testid="stVirtualDropdown"], div[data-baseweb="popover"] div[data-baseweb="menu"] { background-color: var(--surface) !important; }
    ul[data-testid="stVirtualDropdown"] li, div[data-baseweb="popover"] li { color: var(--ink) !important; background-color: var(--surface) !important; }
    ul[data-testid="stVirtualDropdown"] li:hover { background-color: var(--surface-alt) !important; }

    .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"],
    .stSlider label, .stSlider div { color: var(--ink) !important; }
    .stSlider [role="slider"] { background-color: var(--primary) !important; }
    .stSlider [data-testid="stThumbValue"], .stSlider div[data-testid="stTickBar"] { color: var(--ink) !important; }
    div[data-baseweb="slider"] div[role="slider"] div { color: #FFFFFF !important; }
    div[data-baseweb="slider"] > div > div > div { background: var(--primary) !important; }

    .stRadio label, .stRadio div, .stRadio p, .stRadio span { color: var(--ink) !important; }
    .stRadio [role="radiogroup"] { display: flex; gap: 0.4rem; flex-wrap: wrap; }
    .stRadio [role="radiogroup"] label {
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        padding: 0.42rem 0.9rem !important;
        background: var(--surface) !important;
        margin: 0 !important;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .stRadio [role="radiogroup"] label > div:first-child { display: none !important; }
    .stRadio [role="radiogroup"] label p { font-size: 0.85rem !important; font-weight: 600 !important; font-family: 'Sora', sans-serif !important; }
    .stRadio [role="radiogroup"] label:hover { border-color: var(--primary) !important; }
    .stRadio [role="radiogroup"] label:has(input:checked) {
        background: var(--primary) !important;
        border-color: #0F5E45 !important;
    }
    .stRadio [role="radiogroup"] label:has(input:checked) p { color: #FFFFFF !important; }

    div[data-baseweb="select"] > div:hover { border-color: var(--primary) !important; }
    div[data-baseweb="select"]:focus-within > div { border-color: var(--primary) !important; box-shadow: 0 0 0 3px var(--primary-dim) !important; }

    input, textarea {
        background-color: var(--surface) !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
    }
    [data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { background-color: var(--surface) !important; color: var(--ink) !important; }
    [data-testid="stNumberInputStepDown"] svg, [data-testid="stNumberInputStepUp"] svg { fill: var(--ink) !important; }

    [data-testid="stAlert"] p { color: inherit !important; }

    /* =============================================================
       THE PREMIUM CARD — signature element
       ============================================================= */
    .premium-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.4rem 1.5rem 1.3rem 1.5rem;
        margin: 0.2rem 0 1.2rem 0;
        box-shadow: var(--shadow-lg);
    }
    .premium-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1.1rem;
    }
    .premium-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--ink-dim);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.3rem;
    }
    .premium-serial {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem;
        color: var(--ink-dim);
    }
    .risk-pill {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        white-space: nowrap;
    }
    .pill-low { color: var(--primary); background: var(--primary-dim); border: 1px solid var(--primary-line); }
    .pill-moderate { color: var(--amber); background: var(--amber-dim); border: 1px solid rgba(184,122,34,0.35); }
    .pill-high { color: var(--coral); background: var(--coral-dim); border: 1px solid rgba(189,68,56,0.35); }

    .premium-body {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1.2rem;
        flex-wrap: wrap;
    }
    .premium-figure { flex: 1; min-width: 10rem; }
    .premium-amount {
        font-family: 'Sora', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        color: var(--ink);
        line-height: 1.05;
        letter-spacing: -0.02em;
    }
    .premium-amount .unit { font-family: 'Inter', sans-serif; font-weight: 600; font-size: 1rem; color: var(--ink-dim); margin-left: 0.3rem; }
    .premium-caption { font-size: 0.85rem; color: var(--ink-dim); margin-top: 0.3rem; }

    .ring-wrap { display: flex; flex-direction: column; align-items: center; flex-shrink: 0; }
    .ring {
        width: 128px; height: 128px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .ring-inner {
        width: 100px; height: 100px; border-radius: 50%;
        background: var(--surface);
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        box-shadow: inset 0 0 0 1px var(--line);
    }
    .ring-pct { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.25rem; color: var(--ink); }
    .ring-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.58rem; color: var(--ink-dim); text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.1rem; text-align: center; }
    .ring-caption { font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; color: var(--ink-dim); text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.5rem; text-align: center; }

    .premium-footer {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.6rem;
        margin-top: 1.1rem;
        padding-top: 1.1rem;
        border-top: 1px solid var(--line);
    }
    .footer-cell { text-align: left; }
    .footer-k { font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; color: var(--ink-dim) !important; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem; }
    .footer-v { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.05rem; color: var(--ink) !important; }

    .footer-note {
        display: flex; justify-content: space-between;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem; color: var(--ink-dim); margin: 1.8rem 0 0.5rem 0;
    }
    .footer-caption { font-size: 0.75rem; color: var(--ink-dim); text-align: center; line-height: 1.5; padding-bottom: 0.5rem; }

    footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# LOAD MODEL BUNDLE
# ---------------------------------------------------------------
@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_reference_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return None

try:
    bundle = load_bundle()
    scaler = bundle["scaler"]
    poly = bundle["poly"]
    model = bundle["model"]
    columns = bundle["columns"]
except FileNotFoundError:
    cwd = os.getcwd()
    try:
        present = os.listdir(cwd)
    except Exception:
        present = ["(could not list directory)"]
    st.error(
        f"Could not find `{MODEL_PATH}`.\n\n"
        f"**Looking in:** `{cwd}`\n\n"
        f"**Files actually present there:** {present}\n\n"
        f"`{MODEL_PATH}` must be in this exact folder, spelled exactly this way "
        "(check for a `.pkl.pkl` or `(1).pkl` from a browser download, and check it isn't "
        "excluded by a `.gitignore` if deployed from GitHub)."
    )
    st.stop()

ref_df = load_reference_data()

# ---------------------------------------------------------------
# PREDICTION HELPERS
# ---------------------------------------------------------------
def build_row(age, bmi, children, sex, smoker, region):
    smoker_flag = 1 if smoker == "yes" else 0
    row = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "bmi_smoker": [bmi * smoker_flag],
        "obese_smoker": [1 if (bmi >= 30 and smoker == "yes") else 0],
        "sex_male": [sex == "male"],
        "smoker_yes": [smoker == "yes"],
        "region_northwest": [region == "northwest"],
        "region_southeast": [region == "southeast"],
        "region_southwest": [region == "southwest"],
    })
    return row[columns]

def predict(age, bmi, children, sex, smoker, region):
    row = build_row(age, bmi, children, sex, smoker, region)
    row_scaled = scaler.transform(row)
    row_poly = poly.transform(row_scaled)
    return float(model.predict(row_poly)[0])

def make_serial(age, bmi, children, sex, smoker, region):
    key = f"{age}-{bmi}-{children}-{sex}-{smoker}-{region}".encode()
    digest = hashlib.sha256(key).hexdigest()
    digits = "".join(c for c in digest if c.isdigit())[:8].ljust(8, "0")
    return f"CM-{digits[:4]}-{digits[4:]}"

def risk_tier(frac):
    if frac < 0.33:
        return "Low Risk", "pill-low", "tag-low", "var(--primary)"
    elif frac < 0.66:
        return "Moderate Risk", "pill-moderate", "tag-moderate", "var(--amber)"
    return "High Risk", "pill-high", "tag-high", "var(--coral)"

# ---------------------------------------------------------------
# SECTION ICONS (self-drawn, minimal line glyphs)
# ---------------------------------------------------------------
ICON_SUBJECT = '''<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="12" cy="8" r="3.4" stroke="#157A5B" stroke-width="1.8"/>
    <path d="M5 20c0-3.6 3.1-6 7-6s7 2.4 7 6" stroke="#157A5B" stroke-width="1.8" stroke-linecap="round"/>
</svg>'''

ICON_VITALS = '''<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M3 12h4l2-6 4 12 2-6h6" stroke="#157A5B" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>'''

ICON_COVERAGE = '''<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 3l7 3v5c0 4.6-3 8.4-7 10-4-1.6-7-5.4-7-10V6l7-3z" stroke="#157A5B" stroke-width="1.8" stroke-linejoin="round"/>
</svg>'''

# ---------------------------------------------------------------
# BRAND HEADER + HEADING
# ---------------------------------------------------------------
now = datetime.datetime.now()
today = now.strftime("%d %b %Y")
timestamp = now.strftime("%H:%M")

st.markdown(f'''
<div class="brand-bar">
    <div class="brand"><span class="mark"></span>CareMetric</div>
    <div class="status"><span class="dot"></span>MODEL READY · {timestamp}</div>
</div>
<div class="page-heading">
    <div class="eyebrow">Predictive Premium Intelligence</div>
    <h1>Estimate an Annual Premium</h1>
    <div class="sub">Enter the applicant's profile below to generate an instant premium estimate, risk tier, and peer comparison.</div>
</div>
''', unsafe_allow_html=True)

# ---------------------------------------------------------------
# INTAKE — CARD
# ---------------------------------------------------------------
st.markdown('<div class="intake-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Applicant Profile <span class="card-tag">6 FIELDS</span></div>', unsafe_allow_html=True)

def bmi_category(value):
    if value < 18.5:
        return "Underweight", "tag-dim"
    elif value < 25:
        return "Normal", "tag-low"
    elif value < 30:
        return "Overweight", "tag-moderate"
    return "Obese", "tag-high"

st.markdown(f'<div class="section-label">{ICON_SUBJECT} Subject</div>', unsafe_allow_html=True)
id_col1, id_col2 = st.columns(2)
with id_col1:
    with st.container(border=True):
        st.markdown('<div class="field-label">Age (yrs)</div>', unsafe_allow_html=True)
        age = st.slider("Age", 18, 100, 30, label_visibility="collapsed")
with id_col2:
    with st.container(border=True):
        st.markdown('<div class="field-label">Sex</div>', unsafe_allow_html=True)
        sex = st.radio("Sex", ["male", "female"],
                        format_func=lambda v: {"male": "Male", "female": "Female"}[v],
                        horizontal=True, label_visibility="collapsed", key="sex_choice")

st.markdown(f'<div class="section-label">{ICON_VITALS} Vitals</div>', unsafe_allow_html=True)
h_col1, h_col2 = st.columns(2)
with h_col1:
    with st.container(border=True):
        st.markdown('<div class="field-label">BMI</div>', unsafe_allow_html=True)
        bmi = st.slider("BMI", 15.0, 55.0, 25.0, step=0.1, label_visibility="collapsed")
        cat_label, cat_class = bmi_category(bmi)
        st.markdown(f'<span class="field-tag {cat_class}">{cat_label}</span>', unsafe_allow_html=True)
with h_col2:
    with st.container(border=True):
        st.markdown('<div class="field-label">Smoker</div>', unsafe_allow_html=True)
        smoker = st.radio("Smoker", ["no", "yes"],
                           format_func=lambda v: {"no": "No", "yes": "Yes"}[v],
                           horizontal=True, label_visibility="collapsed", key="smoker_choice")

st.markdown(f'<div class="section-label">{ICON_COVERAGE} Coverage</div>', unsafe_allow_html=True)
c_col1, c_col2 = st.columns(2)
with c_col1:
    with st.container(border=True):
        st.markdown('<div class="field-label">Children</div>', unsafe_allow_html=True)
        children = st.slider("Children", 0, 6, 0, label_visibility="collapsed")
with c_col2:
    with st.container(border=True):
        st.markdown('<div class="field-label">Region</div>', unsafe_allow_html=True)
        region = st.selectbox(
            "Region", ["northeast", "northwest", "southeast", "southwest"],
            format_func=lambda v: v.capitalize(),
            label_visibility="collapsed", key="region_choice",
        )

st.markdown('<div style="height: 0.6rem;"></div>', unsafe_allow_html=True)
run = st.button("Calculate Premium", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# RESULTS — PREMIUM CARD + COMPARISONS
# ---------------------------------------------------------------
if run or "last_pred" in st.session_state:

    pred = predict(age, bmi, children, sex, smoker, region)
    st.session_state["last_pred"] = pred

    percentile = (ref_df["charges"] < pred).mean() * 100 if ref_df is not None else None

    if ref_df is not None:
        gauge_max = float(np.ceil(ref_df["charges"].max() / 5000) * 5000)
    else:
        gauge_max = 55000.0

    frac = max(0.0, min(1.0, pred / gauge_max))
    pct_display = round(frac * 100)
    serial = make_serial(age, bmi, children, sex, smoker, region)
    tier_label, pill_class, _, ring_color = risk_tier(frac)

    median_stat = f"${ref_df['charges'].median():,.0f}" if ref_df is not None else "—"
    percentile_stat = f"{percentile:.0f}%" if percentile is not None else "—"
    records_stat = f"{len(ref_df):,}" if ref_df is not None else "—"

    st.markdown(f'''
    <div class="premium-card">
        <div class="premium-head">
            <div>
                <div class="premium-eyebrow">Predicted Annual Premium</div>
                <div class="premium-serial">{serial} · issued {today}</div>
            </div>
            <span class="risk-pill {pill_class}">{tier_label}</span>
        </div>
        <div class="premium-body">
            <div class="premium-figure">
                <div class="premium-amount">${pred:,.0f}<span class="unit">/ year</span></div>
                <div class="premium-caption">Estimated cost for this applicant profile</div>
            </div>
            <div class="ring-wrap">
                <div class="ring" style="background: conic-gradient({ring_color} {pct_display}%, var(--surface-alt) 0);">
                    <div class="ring-inner">
                        <div class="ring-pct">{pct_display}%</div>
                        <div class="ring-label">of range</div>
                    </div>
                </div>
                <div class="ring-caption">Network scale position</div>
            </div>
        </div>
        <div class="premium-footer">
            <div class="footer-cell"><div class="footer-k">Network Median</div><div class="footer-v">{median_stat}</div></div>
            <div class="footer-cell"><div class="footer-k">Percentile</div><div class="footer-v">{percentile_stat}</div></div>
            <div class="footer-cell"><div class="footer-k">Records on File</div><div class="footer-v">{records_stat}</div></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    report = pd.DataFrame([{
        "age": age, "bmi": bmi, "children": children, "sex": sex,
        "smoker": smoker, "region": region,
        "estimated_charge": round(pred, 2),
        "risk_tier": tier_label,
        "serial": serial,
        "date_issued": today,
    }])
    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button("Download Estimate (CSV)", data=csv, file_name="caremetric_estimate.csv",
                        mime="text/csv", use_container_width=True)

else:
    st.markdown('''
    <div class="intake-card" style="text-align:center;">
        <div class="card-title" style="justify-content:center; border-bottom:none;">No Estimate Yet</div>
        <div style="color:var(--ink-dim); font-size:0.9rem;">Fill in the applicant profile above, then select
        <strong>Calculate Premium</strong> to generate your CareMetric card.</div>
    </div>
    ''', unsafe_allow_html=True)

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown(f'''
    <div class="footer-note">
        <span>CAREMETRIC — PREDICTIVE PREMIUM INTELLIGENCE</span>
        <span>{today}</span>
    </div>
    <div class="footer-caption">
        Model: ridge regression with degree-2 polynomial features, trained on the classic
        medical insurance charges dataset.<br>Not medical or financial advice.
    </div>
''', unsafe_allow_html=True)
