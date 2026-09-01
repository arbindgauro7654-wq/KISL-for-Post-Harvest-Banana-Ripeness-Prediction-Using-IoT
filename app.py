"""RipeSense - Knowledge-Integrated Banana Ripeness Decision-Support App.

An interactive Streamlit dashboard that visualises the data, the knowledge
graph, model performance, interpretability and robustness, and provides a live
prediction tool. Start it with:  py run_app.py
(or directly:  py -m streamlit run app.py)
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src import config as C
from src.decision_support import predict_compare, predict_one
from src.kg_features import KGFeatureGenerator

# --------------------------------------------------------------------------- #
# Page config & theme
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="RipeSense - Banana Ripeness Decision Support",
                   page_icon=":material/eco:", layout="wide",
                   initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
    :root {
        --bg:#fbf7ee; --surface:#ffffff; --ink:#241e14; --muted:#6b6153;
        --line:#e8e2d5; --accent:#e3a008; --accent-ink:#a3730a;
        --accent-weak:rgba(227,160,8,0.13);
        --leaf:#2f7a45; --olive:#7c8c22; --gold:#c9a227; --peel:#d98324;
        --brown:#8a5a2f; --cream:#fdf3d8;
        --radius:10px;
        --shadow:0 1px 2px rgba(60,45,20,0.05), 0 4px 14px rgba(60,45,20,0.05);
        --font:"Inter","Segoe UI",system-ui,-apple-system,"Helvetica Neue",sans-serif;
    }

    /* ---------------- Base ---------------- */
    .stApp { background:var(--bg); font-family:var(--font); }
    .stApp p, .stApp li, .stApp label, .stApp button, .stApp input,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family:var(--font); }
    /* Never override Streamlit's icon font, or ligatures leak as raw text. */
    [data-testid="stIconMaterial"], .material-symbols-rounded,
    span[class*="material-symbols"] {
        font-family:"Material Symbols Rounded" !important; }
    section[data-testid="stMain"] .block-container {
        padding-top:2.6rem; padding-bottom:3.5rem; max-width:1340px; }

    /* ---------------- Typography ---------------- */
    section[data-testid="stMain"] h2 { color:var(--ink); font-size:1.3rem;
        font-weight:640; letter-spacing:-0.012em; margin:1.6rem 0 0.4rem 0; }
    section[data-testid="stMain"] h3 { color:var(--ink); font-size:1rem;
        font-weight:640; letter-spacing:-0.005em; margin:1.9rem 0 0.7rem 0;
        padding-bottom:0.5rem; border-bottom:1px solid var(--line); }
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p {
        color:#33403a; font-size:0.94rem; line-height:1.62; }
    section[data-testid="stMain"] [data-testid="stCaptionContainer"] p {
        color:var(--muted); font-size:0.85rem; font-weight:450; line-height:1.55; }
    section[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stMain"] [data-testid="stWidgetLabel"] label,
    section[data-testid="stMain"] [role="radiogroup"] label {
        color:#2a3831; font-weight:550; font-size:0.87rem; }
    section[data-testid="stMain"] [data-testid="stExpander"] summary p {
        color:var(--ink); font-weight:600; font-size:0.9rem; }
    section[data-testid="stMain"] [data-testid="stExpander"] details {
        border:1px solid var(--line); border-radius:var(--radius);
        background:var(--surface); }
    section[data-testid="stMain"] code { background:var(--accent-weak) !important;
        color:var(--accent-ink) !important; font-size:0.86em; }

    /* ---------------- Page header ---------------- */
    .rs-page { display:flex; align-items:flex-start; gap:0.95rem;
        padding-bottom:1.15rem; margin-bottom:1.5rem;
        border-bottom:1px solid var(--line); }
    .rs-page .glyph { flex:none; width:42px; height:42px; border-radius:9px;
        background:var(--accent-weak); color:var(--accent); display:flex;
        align-items:center; justify-content:center; }
    /* Overview variant: warm ripening gradient and a stage ramp on the right. */
    .rs-page.hero { position:relative; overflow:hidden; border-bottom:none;
        align-items:center; padding:1.4rem 1.6rem; margin-bottom:1.6rem;
        border:1px solid #f0e3c2; border-radius:14px;
        background:linear-gradient(100deg,#ffffff 0%,#fffaee 42%,#fdf1d2 100%);
        box-shadow:0 1px 2px rgba(60,45,20,0.05), 0 8px 24px rgba(190,150,40,0.09); }
    .rs-page.hero:before { content:""; position:absolute; left:0; top:0; bottom:0;
        width:4px; background:linear-gradient(180deg,var(--leaf) 0%,var(--olive) 28%,
            var(--gold) 55%,var(--accent) 78%,var(--brown) 100%); }
    .rs-page.hero .glyph { width:48px; height:48px; background:#ffffff;
        border:1px solid #f0e3c2; color:var(--accent);
        box-shadow:0 2px 8px rgba(190,150,40,0.15); }
    .rs-page.hero .ttl { font-size:1.72rem; }
    .rs-ramp { margin-left:auto; padding-left:1.5rem; text-align:right; flex:none; }
    .rs-ramp .dots { display:flex; gap:6px; justify-content:flex-end;
        margin-bottom:0.4rem; }
    .rs-ramp .dots i { width:16px; height:16px; border-radius:50%;
        border:2px solid #ffffff; box-shadow:0 1px 4px rgba(60,45,20,0.16); }
    .rs-ramp .cap { font-size:0.63rem; font-weight:700; letter-spacing:0.1em;
        text-transform:uppercase; color:#9a8a68; }
    .rs-page .eyebrow { font-size:0.67rem; font-weight:700; letter-spacing:0.11em;
        text-transform:uppercase; color:var(--accent); margin-bottom:0.22rem; }
    .rs-page .ttl { font-size:1.48rem; font-weight:650; color:var(--ink);
        letter-spacing:-0.022em; line-height:1.2; }
    .rs-page .sub { margin-top:0.34rem; color:var(--muted); font-size:0.9rem;
        line-height:1.55; max-width:80ch; }

    /* ---------------- Metric cards ---------------- */
    .metric-card { position:relative; overflow:hidden; background:var(--surface);
        border:1px solid var(--line); border-radius:var(--radius);
        padding:0.9rem 1rem 0.95rem 1.15rem; box-shadow:var(--shadow);
        transition:transform 0.16s ease, box-shadow 0.16s ease; }
    .metric-card:before { content:""; position:absolute; left:0; top:0; bottom:0;
        width:4px; background:var(--tone,var(--accent)); }
    .metric-card:after { content:""; position:absolute; right:-28px; top:-28px;
        width:84px; height:84px; border-radius:50%; opacity:0.09;
        background:var(--tone,var(--accent)); }
    .metric-card:hover { transform:translateY(-2px);
        box-shadow:0 2px 4px rgba(60,45,20,0.06), 0 10px 24px rgba(60,45,20,0.08); }
    .metric-card .lb { display:block; color:var(--muted); font-size:0.68rem;
        font-weight:700; text-transform:uppercase; letter-spacing:0.09em; }
    .metric-card .vl { margin-top:0.4rem; font-size:1.55rem; font-weight:640;
        color:var(--tone,var(--ink)); letter-spacing:-0.025em;
        font-variant-numeric:tabular-nums; }

    .pill { display:inline-block; background:var(--accent-weak); color:var(--accent-ink);
        padding:0.14rem 0.5rem; border-radius:5px; font-size:0.71rem; font-weight:650;
        letter-spacing:0.02em; margin-right:0.45rem; }
    .rulebox, .firedrule { background:var(--surface); border:1px solid var(--line);
        border-left:3px solid var(--tone,var(--accent)); border-radius:var(--radius);
        padding:0.7rem 0.9rem; margin-bottom:0.5rem; color:#413828;
        font-size:0.91rem; line-height:1.55; box-shadow:var(--shadow);
        animation:rsIn 0.24s ease-out both; }
    .rulebox b, .rulebox strong { color:var(--accent-ink); font-weight:650; }

    /* ---------------- Live prediction ---------------- */
    .rs-head { display:flex; align-items:center; gap:0.6rem; margin:1.8rem 0 0.8rem 0;
        font-size:0.72rem; font-weight:700; letter-spacing:0.11em;
        text-transform:uppercase; color:var(--muted); }
    .rs-head .rule { flex:1; height:1px; background:var(--line); }
    .rs-live { display:inline-flex; align-items:center; gap:0.36rem;
        color:var(--accent); font-size:0.67rem; font-weight:700; letter-spacing:0.11em; }
    .rs-live .dot { width:6px; height:6px; border-radius:50%; background:var(--accent); }

    /* Bordered containers hold the sensor groups, so the sliders sit inside. */
    div[data-testid="stVerticalBlockBorderWrapper"] { background:var(--surface);
        border:1px solid var(--line) !important; border-radius:var(--radius) !important;
        box-shadow:var(--shadow); }
    div[data-testid="stVerticalBlockBorderWrapper"] > div { border:none !important; }
    .sensor-group-title { display:flex; align-items:center; gap:0.5rem;
        color:var(--ink); font-weight:640; font-size:0.88rem; margin:0 0 0.4rem 0;
        padding-bottom:0.6rem; border-bottom:1px solid var(--line); }
    .sensor-group-title svg { color:var(--accent); }

    .rs-gauge { background:var(--surface); border:1px solid var(--line);
        border-radius:var(--radius); padding:0.8rem 0.8rem 0.7rem;
        box-shadow:var(--shadow); animation:rsIn 0.26s ease-out both;
        transition:border-color 0.15s ease; }
    .rs-gauge:hover { border-color:var(--tone,var(--accent)); }
    .rs-gauge .nm { display:flex; align-items:flex-start; gap:0.35rem;
        min-height:2.1em; color:var(--muted); font-size:0.66rem; font-weight:700;
        letter-spacing:0.07em; text-transform:uppercase; line-height:1.25; }
    .rs-gauge .nm svg { color:var(--tone,var(--accent)); flex:none; }
    .rs-gauge .vl { color:var(--ink); font-weight:640; font-size:1.32rem;
        margin:0.5rem 0 0.55rem 0; letter-spacing:-0.025em;
        font-variant-numeric:tabular-nums; }
    .rs-gauge .un { font-size:0.7rem; font-weight:600; color:var(--tone,var(--muted));
        margin-left:0.15rem; }
    .rs-track { height:5px; border-radius:3px; background:#f0ebe0; overflow:hidden; }
    .rs-fill { height:100%; border-radius:3px; background:var(--tone,var(--accent));
        animation:rsFill 0.38s cubic-bezier(0.22,0.8,0.3,1) both; }
    .rs-range { display:flex; justify-content:space-between; font-size:0.63rem;
        color:#a2977f; font-weight:600; margin-top:0.32rem;
        font-variant-numeric:tabular-nums; }

    .rs-strip { display:flex; gap:0.5rem; margin:0 0 1rem 0; }
    .rs-stage { flex:1; text-align:center; background:var(--surface);
        border:1px solid var(--line); border-radius:var(--radius);
        padding:0.75rem 0.5rem; animation:rsIn 0.26s ease-out both;
        transition:border-color 0.18s ease, transform 0.18s ease,
                   box-shadow 0.18s ease; }
    .rs-stage .sw { display:block; width:12px; height:12px; border-radius:50%;
        margin:0 auto 0.5rem; opacity:0.45; }
    .rs-stage .st { font-size:0.7rem; font-weight:700; color:var(--muted);
        letter-spacing:0.06em; text-transform:uppercase; }
    .rs-stage .lb { font-size:0.67rem; color:#a2977f; font-weight:500;
        line-height:1.3; margin-top:0.22rem; }
    .rs-stage.active { border-color:var(--sc); transform:translateY(-3px);
        box-shadow:0 3px 14px var(--glow); }
    .rs-stage.active .sw { opacity:1; transform:scale(1.25);
        box-shadow:0 0 0 4px var(--glow); }
    .rs-stage.active .st { color:var(--sc); }
    .rs-stage.active .lb { color:var(--muted); }

    .rs-conf-wrap { background:var(--surface); border:1px solid var(--line);
        border-radius:var(--radius); padding:0.75rem 0.9rem 0.8rem;
        box-shadow:var(--shadow); margin-bottom:1.3rem;
        animation:rsIn 0.26s ease-out both; }
    .rs-conf { height:7px; border-radius:4px; background:#eaefeb; overflow:hidden; }
    .rs-conf-fill { height:100%; border-radius:4px;
        animation:rsFill 0.42s cubic-bezier(0.22,0.8,0.3,1) both; }
    .rs-conf-cap { display:flex; justify-content:space-between; align-items:baseline;
        margin-bottom:0.5rem; font-size:0.74rem; color:var(--muted); font-weight:700;
        letter-spacing:0.06em; text-transform:uppercase; }
    .rs-conf-cap b { color:var(--ink); font-weight:650; font-size:0.95rem;
        letter-spacing:0; font-variant-numeric:tabular-nums; }

    .stage-banner { display:flex; align-items:center; gap:0.9rem;
        background:var(--surface); border:1px solid var(--line);
        border-left:3px solid var(--sc); border-radius:var(--radius);
        padding:1rem 1.15rem; box-shadow:var(--shadow);
        animation:rsIn 0.28s ease-out both; }
    .stage-banner .sw { flex:none; width:34px; height:34px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        border:2px solid var(--sc); }
    .stage-banner .sw i { display:block; width:14px; height:14px; border-radius:50%;
        background:var(--sc); }
    .stage-banner .t { font-size:1.05rem; font-weight:650; color:var(--ink);
        letter-spacing:-0.015em; }
    .stage-banner small { display:block; font-size:0.81rem; font-weight:500;
        color:var(--muted); margin-top:0.22rem; }

    /* ---------------- Widgets ---------------- */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border:1px solid var(--line); border-radius:var(--radius); overflow:hidden; }
    .stButton > button { border-radius:7px; font-weight:600; padding:0.5rem 1.15rem;
        box-shadow:none; }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background:var(--accent); border:1px solid var(--accent); }
    .stButton > button[kind="primary"] p,
    .stButton > button[data-testid="stBaseButton-primary"] p,
    .stButton > button[kind="primary"] div,
    .stButton > button[data-testid="stBaseButton-primary"] div {
        color:#ffffff !important; }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background:var(--accent-ink); border-color:var(--accent-ink); }
    div[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line);
        border-radius:var(--radius); padding:0.75rem 0.9rem; box-shadow:var(--shadow); }
    div[data-testid="stAlert"], div[data-testid="stAlertContainer"] {
        border-radius:var(--radius); background:var(--surface) !important;
        border:1px solid var(--line); box-shadow:var(--shadow); }
    div[data-testid="stAlert"] p, div[data-testid="stAlertContainer"] p {
        color:#33403a !important; font-size:0.9rem; }
    div[data-testid="stAlert"] svg, div[data-testid="stAlertContainer"] svg {
        fill:var(--accent) !important; color:var(--accent) !important; }

    /* ---------------- Motion ---------------- */
    @keyframes rsIn { from { opacity:0; transform:translateY(6px); }
                      to   { opacity:1; transform:none; } }
    @keyframes rsFill { from { width:0; } }
    @media (prefers-reduced-motion: reduce) {
        * { animation:none !important; transition:none !important; } }
    /* ---------------- Sidebar ---------------- */
    section[data-testid="stSidebar"] { background:#231d13; border-right:1px solid #372f21; }
    section[data-testid="stSidebar"] * { color:#e4dccc; }
    section[data-testid="stSidebar"] .block-container { padding-top:1.8rem; }
    section[data-testid="stSidebar"] hr { border-color:#372f21; margin:1.1rem 0; }
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size:0.66rem; letter-spacing:0.12em; text-transform:uppercase;
        color:#a3947a; font-weight:700; }
    section[data-testid="stSidebar"] [role="radiogroup"] label p {
        font-size:0.88rem; font-weight:500; color:#e4dccc; }
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover p {
        color:#ffffff; }
    .rs-brand { display:flex; align-items:center; gap:0.65rem; padding-bottom:1.1rem;
        border-bottom:1px solid #372f21; margin-bottom:0.4rem; }
    .rs-brand .mark { flex:none; width:34px; height:34px; border-radius:8px;
        background:linear-gradient(140deg,#3a3018,#2b2416); border:1px solid #4a3d23;
        color:var(--gold); display:flex; align-items:center; justify-content:center; }
    .rs-brand .nm { font-size:1rem; font-weight:650; color:#fbf6ea;
        letter-spacing:-0.015em; line-height:1.2; }
    .rs-brand .tg { font-size:0.63rem; color:#a3947a; letter-spacing:0.09em;
        text-transform:uppercase; margin-top:0.15rem; }
    .rs-status { border:1px solid #372f21; border-radius:8px; padding:0.65rem 0.75rem;
        background:#2a2317; }
    .rs-status .k { display:block; font-size:0.6rem; color:#a3947a; font-weight:700;
        letter-spacing:0.11em; text-transform:uppercase; }
    .rs-status .v { font-size:0.83rem; color:#f3ecdd; font-weight:600;
        margin-top:0.18rem; display:block; }
    .rs-status .ok { color:#8dc98a; }
    .rs-status .bad { color:#e0906f; }
    /* Ripening ramp strip under the brand mark */
    .rs-scale { display:flex; height:4px; border-radius:2px; overflow:hidden;
        margin:0.9rem 0 0.2rem; }
    .rs-scale i { flex:1; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

INK, MUTED, LINE = "#241e14", "#6b6153", "#e8e2d5"
ACCENT, ACCENT_SOFT, NEUTRAL = "#e3a008", "#f0c968", "#c7bfae"

# The banana ripening ramp drives the whole palette: leaf green, olive, gold,
# banana yellow, peel brown.
BANANA = {"leaf": "#2f7a45", "olive": "#7c8c22", "gold": "#c9a227",
          "amber": "#e3a008", "peel": "#d98324", "brown": "#8a5a2f",
          "teal": "#17756b"}
CARD_TONES = [BANANA["leaf"], BANANA["gold"], BANANA["amber"], BANANA["brown"],
              BANANA["olive"], BANANA["teal"]]

# Each page carries its own accent, drawn from the same ripening ramp.
PAGE_ACCENT = {
    "System Architecture": BANANA["teal"],
    "Overview": BANANA["amber"],
    "Data Explorer": BANANA["leaf"],
    "Knowledge Graph": BANANA["olive"],
    "Model Results": BANANA["peel"],
    "Interpretability": BANANA["brown"],
    "Robustness": BANANA["teal"],
    "Decision Support": BANANA["amber"],
}

# Ripeness stage colours: green through to brown.
STAGE_COLORS = {1: "#2f7a45", 2: "#7c9c34", 3: "#c9a227", 4: "#e3a008", 5: "#8a5a2f"}
# Darker variants for text on a white background.
STAGE_TEXT = {1: "#215a33", 2: "#5b7326", 3: "#93741a", 4: "#a3730a", 5: "#6b4423"}

# Friendly label, unit and icon key for each sensor feature.
FEATURE_META = {
    "Temp-int": ("Internal Temperature", "°C", "temperature"),
    "Humid-int": ("Internal Humidity", "%RH", "humidity"),
    "Press-int": ("Internal Pressure", "hPa", "pressure"),
    "Temp-ext": ("Ambient Temperature", "°C", "temperature"),
    "Humid-ext": ("Ambient Humidity", "%RH", "humidity"),
    "Press-ext": ("Ambient Pressure", "hPa", "pressure"),
}

# One colour per sensor type, so the six live gauges read at a glance.
SENSOR_TONES = {"temperature": BANANA["peel"], "humidity": BANANA["teal"],
                "pressure": BANANA["olive"]}

# --------------------------------------------------------------------------- #
# Inline SVG icon set (stroked, 24x24, inherits colour from its container)
# --------------------------------------------------------------------------- #
ICON_PATHS = {
    "overview": ("<rect x='3' y='3' width='7' height='9' rx='1.2'/>"
                 "<rect x='14' y='3' width='7' height='5' rx='1.2'/>"
                 "<rect x='14' y='12' width='7' height='9' rx='1.2'/>"
                 "<rect x='3' y='16' width='7' height='5' rx='1.2'/>"),
    "data": ("<line x1='18' y1='20' x2='18' y2='10'/>"
             "<line x1='12' y1='20' x2='12' y2='4'/>"
             "<line x1='6' y1='20' x2='6' y2='14'/>"),
    "graph": ("<circle cx='18' cy='5' r='2.6'/><circle cx='6' cy='12' r='2.6'/>"
              "<circle cx='18' cy='19' r='2.6'/>"
              "<line x1='8.3' y1='13.4' x2='15.7' y2='17.6'/>"
              "<line x1='15.7' y1='6.4' x2='8.3' y2='10.6'/>"),
    "results": ("<circle cx='12' cy='12' r='9'/><circle cx='12' cy='12' r='5'/>"
                "<circle cx='12' cy='12' r='1.4'/>"),
    "interpret": ("<circle cx='11' cy='11' r='7'/>"
                  "<line x1='21' y1='21' x2='16.2' y2='16.2'/>"
                  "<line x1='8.6' y1='12.6' x2='8.6' y2='9.4'/>"
                  "<line x1='11' y1='13.6' x2='11' y2='8.4'/>"
                  "<line x1='13.4' y1='13.6' x2='13.4' y2='11'/>"),
    "robust": ("<path d='M12 21.2s7.4-3.5 7.4-9.3V5.6L12 2.8 4.6 5.6v6.3"
               "c0 5.8 7.4 9.3 7.4 9.3z'/><polyline points='9.2 11.8 11.4 14 15 10'/>"),
    "live": "<polyline points='22 12 18 12 15 21 9 3 6 12 2 12'/>",
    "temperature": ("<path d='M14 14.8V3.5a2.5 2.5 0 0 0-5 0v11.3a4.5 4.5 0 1 0 5 0z'/>"
                    "<line x1='11.5' y1='7.5' x2='11.5' y2='15.5'/>"),
    "humidity": "<path d='M12 2.7l5.7 5.7a8 8 0 1 1-11.4 0z'/>",
    "pressure": ("<path d='M4.2 17.6a9 9 0 1 1 15.6 0'/>"
                 "<line x1='12' y1='17.2' x2='15.6' y2='11.4'/>"
                 "<circle cx='12' cy='17.8' r='1.3'/>"),
    "enclosure": ("<path d='M20.5 7.8 12 3 3.5 7.8v8.4L12 21l8.5-4.8z'/>"
                  "<polyline points='3.5 7.8 12 12.6 20.5 7.8'/>"
                  "<line x1='12' y1='12.6' x2='12' y2='21'/>"),
    "ambient": ("<circle cx='12' cy='12' r='4'/><line x1='12' y1='2' x2='12' y2='4.2'/>"
                "<line x1='12' y1='19.8' x2='12' y2='22'/>"
                "<line x1='4.2' y1='4.2' x2='5.8' y2='5.8'/>"
                "<line x1='18.2' y1='18.2' x2='19.8' y2='19.8'/>"
                "<line x1='2' y1='12' x2='4.2' y2='12'/>"
                "<line x1='19.8' y1='12' x2='22' y2='12'/>"
                "<line x1='4.2' y1='19.8' x2='5.8' y2='18.2'/>"
                "<line x1='18.2' y1='5.8' x2='19.8' y2='4.2'/>"),
    "readings": ("<rect x='5' y='4.4' width='14' height='16.6' rx='2'/>"
                 "<path d='M9.2 4.4V3.6A1.6 1.6 0 0 1 10.8 2h2.4a1.6 1.6 0 0 1 1.6 1.6v.8'/>"
                 "<line x1='8.8' y1='10' x2='15.2' y2='10'/>"
                 "<line x1='8.8' y1='13.4' x2='15.2' y2='13.4'/>"
                 "<line x1='8.8' y1='16.8' x2='12.6' y2='16.8'/>"),
    "ripeness": ("<path d='M11 20.6A7.6 7.6 0 0 1 9.8 5.9C15.4 4.8 17 4.3 19 1.8"
                 "c1 2 2 4.2 2 8 0 5.8-4.8 10.8-10 10.8z'/>"
                 "<path d='M2.6 21.6c0-3.2 1.9-5.7 5.2-6.4'/>"),
    "advice": ("<path d='M18 8.6a6 6 0 0 0-12 0c0 6.4-2.6 8.4-2.6 8.4h17.2S18 15 18 8.6z'/>"
               "<path d='M13.7 20.6a2 2 0 0 1-3.4 0'/>"),
    "rules": ("<line x1='4' y1='21' x2='4' y2='14'/><line x1='4' y1='10' x2='4' y2='3'/>"
              "<line x1='12' y1='21' x2='12' y2='12'/><line x1='12' y1='8' x2='12' y2='3'/>"
              "<line x1='20' y1='21' x2='20' y2='16'/><line x1='20' y1='12' x2='20' y2='3'/>"
              "<line x1='1.5' y1='14' x2='6.5' y2='14'/>"
              "<line x1='9.5' y1='8' x2='14.5' y2='8'/>"
              "<line x1='17.5' y1='16' x2='22.5' y2='16'/>"),
}


def icon(name: str, size: int = 20, stroke: float = 1.6) -> str:
    return (f"<svg width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' "
            f"stroke='currentColor' stroke-width='{stroke}' stroke-linecap='round' "
            f"stroke-linejoin='round' aria-hidden='true' "
            f"style='display:block;flex:none'>{ICON_PATHS[name]}</svg>")


# --------------------------------------------------------------------------- #
# Artefact loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_results():
    path = os.path.join(C.RESULT_DIR, "model_results.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_json(name):
    path = os.path.join(C.RESULT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


#: Artefacts the Live Prediction page needs before it can run inference.
REQUIRED_ARTEFACTS = ("scaler.pkl", "kg_generator.json", "baseline_rf.pkl",
                      "kg_rf.pkl")


def missing_artefacts() -> list[str]:
    """Names of model files that are absent, so the UI can explain rather than
    raise. They are regenerated by `py -m src.run_pipeline`."""
    return [f for f in REQUIRED_ARTEFACTS
            if not os.path.exists(os.path.join(C.MODEL_DIR, f))]


@st.cache_resource(show_spinner=False)
def load_models():
    scaler = joblib.load(os.path.join(C.MODEL_DIR, "scaler.pkl"))
    gen = KGFeatureGenerator.load(os.path.join(C.MODEL_DIR, "kg_generator.json"))
    models = {}
    for n in ("baseline_rf", "kg_rf", "baseline_xgb", "kg_xgb", "best_model"):
        p = os.path.join(C.MODEL_DIR, f"{n}.pkl")
        if os.path.exists(p):
            models[n] = joblib.load(p)
    return scaler, gen, models


def fig_path(name):
    p = os.path.join(C.FIG_DIR, name)
    return p if os.path.exists(p) else None


@st.cache_data(show_spinner=False)
def load_test_samples(n: int = 40) -> pd.DataFrame:
    """Sample rows from the held-out test CSV for offline simulation."""
    x_path = os.path.join(C.DATA_DIR, "ds_34_x_test.csv")
    y_path = os.path.join(C.DATA_DIR, "ds_34_y_test.csv")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        return pd.DataFrame()
    X = pd.read_csv(x_path, index_col=0)[C.SENSOR_FEATURES]
    y = pd.read_csv(y_path, index_col=0).iloc[:, 0].astype(int)
    df = X.copy()
    df[C.LABEL_NAME] = y.values
    df.insert(0, "row_id", df.index.astype(str))
    step = max(1, len(df) // n)
    return df.iloc[::step].head(n).reset_index(drop=True)


def card(col, label, value, tone: int = 0):
    """Metric tile; `tone` picks a colour from the ripening ramp."""
    colour = CARD_TONES[tone % len(CARD_TONES)]
    col.markdown(
        f"<div class='metric-card' style='--tone:{colour}'>"
        f"<span class='lb'>{label}</span>"
        f"<div class='vl'>{value}</div></div>",
        unsafe_allow_html=True)


def page_header(icon_name: str, title: str, subtitle: str = "",
                eyebrow: str = "", hero: bool = False) -> None:
    """Consistent page banner: glyph, optional eyebrow, title and description."""
    cls = "rs-page hero" if hero else "rs-page"
    parts = [f"<div class='{cls}'><div class='glyph'>{icon(icon_name, 22)}</div><div>"]
    if eyebrow:
        parts.append(f"<div class='eyebrow'>{eyebrow}</div>")
    parts.append(f"<div class='ttl'>{title}</div>")
    if subtitle:
        parts.append(f"<div class='sub'>{subtitle}</div>")
    parts.append("</div>")
    if hero:
        dots = "".join(f"<i style='background:{STAGE_COLORS[s]}'></i>"
                       for s in sorted(STAGE_COLORS))
        parts.append(f"<div class='rs-ramp'><div class='dots'>{dots}</div>"
                     f"<div class='cap'>Green &rarr; Over-ripe</div></div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def shade(hex_color: str, factor: float) -> str:
    """Darken (factor < 1) or lighten (factor > 1) a hex colour."""
    h = hex_color.lstrip("#")
    vals = [min(255, max(0, round(int(h[i:i + 2], 16) * factor))) for i in (0, 2, 4)]
    return "#{:02x}{:02x}{:02x}".format(*vals)


def live_badge(text: str) -> str:
    return (f"<div class='rs-head'><span>{text}</span><span class='rule'></span>"
            f"<span class='rs-live'><span class='dot'></span>Live</span></div>")


def sensor_gauge(feat: str, value: float, lo: float, hi: float,
                 delay: float = 0.0) -> str:
    """Card echoing one sensor value and its position within the observed range."""
    name, unit, icon_name = FEATURE_META[feat]
    tone = SENSOR_TONES[icon_name]
    span = hi - lo
    pct = 0.0 if span <= 0 else max(0.0, min(1.0, (value - lo) / span)) * 100
    return (f"<div class='rs-gauge' style='--tone:{tone};"
            f"animation-delay:{delay:.2f}s'>"
            f"<div class='nm'>{icon(icon_name, 14)}<span>{name}</span></div>"
            f"<div class='vl'>{value:.1f}<span class='un'>{unit}</span></div>"
            f"<div class='rs-track'><div class='rs-fill' "
            f"style='width:{pct:.1f}%;animation-delay:{delay + 0.06:.2f}s'></div></div>"
            f"<div class='rs-range'><span>{lo:.0f}</span><span>{hi:.0f}</span></div>"
            f"</div>")


def stage_strip(stage: int) -> str:
    """Five-stage strip in which the predicted stage is highlighted."""
    cells = []
    for s in C.RIPENESS_STAGES:
        colour = STAGE_COLORS[s]
        label = C.STAGE_LABELS[s].split(" - ")[1]
        active = " active" if s == stage else ""
        border = f"border-color:{colour};" if active else ""
        cells.append(
            f"<div class='rs-stage{active}' style='{border}--sc:{colour};"
            f"--glow:{rgba(colour, 0.22)};animation-delay:{0.04 * s:.2f}s'>"
            f"<span class='sw' style='background:{colour}'></span>"
            f"<div class='st'>Stage {s}</div>"
            f"<div class='lb'>{label}</div></div>")
    return f"<div class='rs-strip'>{''.join(cells)}</div>"


def confidence_meter(confidence: float, colour: str, caption: str) -> str:
    return (f"<div class='rs-conf-wrap'>"
            f"<div class='rs-conf-cap'><span>{caption}</span>"
            f"<b>{confidence * 100:.1f}%</b></div>"
            f"<div class='rs-conf'><div class='rs-conf-fill' "
            f"style='width:{confidence * 100:.1f}%;background:{colour}'></div></div>"
            f"</div>")


# --------------------------------------------------------------------------- #
# Chart theme - applied to every Plotly figure in the app
# --------------------------------------------------------------------------- #
_AXIS = dict(gridcolor="#f0eade", zerolinecolor="#e8e2d5", linecolor="#e0d8c8",
             ticks="outside", tickcolor="#e0d8c8", ticklen=4,
             title_font=dict(size=12, color=MUTED), tickfont=dict(size=11, color=MUTED))
pio.templates["ripesense"] = go.layout.Template(layout=dict(
    font=dict(family='Inter, "Segoe UI", system-ui, sans-serif', size=12, color=INK),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[BANANA["leaf"], BANANA["amber"], BANANA["teal"], BANANA["peel"],
              BANANA["olive"], BANANA["brown"]],
    title=dict(font=dict(size=14, color=INK), x=0, xanchor="left", y=0.97),
    xaxis=_AXIS, yaxis=_AXIS,
    margin=dict(t=48, b=44, l=52, r=24),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                font=dict(size=11, color=MUTED), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(color="#ffffff", size=12)),
    colorscale=dict(sequential=[[0, "#fdf3d8"], [1, BANANA["brown"]]]),
))
pio.templates.default = "ripesense"


RESULTS = load_results()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
_ramp = "".join(f"<i style='background:{STAGE_COLORS[s]}'></i>"
                for s in sorted(STAGE_COLORS))
st.sidebar.markdown(
    f"<div class='rs-brand'><div class='mark'>{icon('ripeness', 18)}</div><div>"
    f"<div class='nm'>RipeSense</div>"
    f"<div class='tg'>Ripeness Monitoring</div></div></div>"
    f"<div class='rs-scale'>{_ramp}</div>",
    unsafe_allow_html=True)
PAGE = st.sidebar.radio(
    "Navigate",
    ["System Architecture", "Overview", "Data Explorer", "Knowledge Graph",
     "Model Results", "Interpretability", "Robustness", "Decision Support"],
    key="nav_page",
)
st.sidebar.markdown("---")

# Each page recolours the shared accent variables, so cards, pills, buttons and
# the header glyph all shift to that section's colour.
ACCENT = PAGE_ACCENT.get(PAGE, BANANA["amber"])
st.markdown(f"<style>:root{{--accent:{ACCENT};"
            f"--accent-ink:{shade(ACCENT, 0.72)};"
            f"--accent-weak:{rgba(ACCENT, 0.13)};}}</style>",
            unsafe_allow_html=True)

if RESULTS:
    bm = RESULTS["best_model"]["name"]
    st.sidebar.markdown(
        f"<div class='rs-status'><span class='k'>Pipeline</span>"
        f"<span class='v ok'>Loaded</span>"
        f"<span class='k' style='margin-top:0.5rem'>Best model</span>"
        f"<span class='v'>{bm}</span></div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown(
        "<div class='rs-status'><span class='k'>Pipeline</span>"
        "<span class='v bad'>Not run</span>"
        "<span class='k' style='margin-top:0.5rem'>Command</span>"
        "<span class='v'>py -m src.run_pipeline</span></div>",
        unsafe_allow_html=True)

if RESULTS is None:
    page_header("ripeness", "RipeSense",
                "No results found yet. Run <code>py -m src.run_pipeline</code> to "
                "generate the model outputs, then reload this page.",
                eyebrow="Knowledge-Integrated Ripeness Monitoring")
    st.stop()


# --------------------------------------------------------------------------- #
# Page: System Architecture
# --------------------------------------------------------------------------- #
if PAGE == "System Architecture":
    page_header(
        "overview", "System Architecture",
        "Integrated AI system: data management, knowledge-graph feature generation, "
        "trained models, evaluation artefacts, and offline decision-support UI.",
        eyebrow="Five-layer design")
    layers = [
        ("Data layer", "ds_34 CSVs, range validation, min–max scaler",
         "data_loader.py", "scaler.pkl, data_report.json"),
        ("Knowledge layer", "Literature triples → flags, risk scores, violation count",
         "knowledge_graph.py, kg_features.py", "kg_generator.json, validated_rules.json"),
        ("Modelling layer", "RF / XGBoost four-model ablation, GridSearchCV",
         "train.py", "baseline_*.pkl, kg_*.pkl, best_model.pkl"),
        ("Evaluation layer", "Metrics, McNemar, SHAP, robustness harness",
         "evaluate.py, robustness.py", "model_results.json, figures/"),
        ("Application layer", "Streamlit decision support, compare & explain",
         "app.py, decision_support.py", "Interactive inference on saved artefacts"),
    ]
    arch = """
    digraph {
      bgcolor="transparent"; rankdir=TB; pad=0.3; nodesep=0.5;
      node [shape=box style="rounded,filled" fontname="Segoe UI" fontsize=10
            penwidth=1.2 color="#e0d8c8" fontcolor="#241e14"];
      edge [color="#c7bfae" penwidth=1.2 arrowsize=0.7];
      D [label="Data\\n(CSV + scaler)" fillcolor="#ffffff"];
      K [label="Knowledge\\n(KG features)" fillcolor="#eef3e4"];
      M [label="Modelling\\n(RF / XGBoost)" fillcolor="#fdf3d8"];
      E [label="Evaluation\\n(metrics / SHAP)" fillcolor="#f5efe4"];
      A [label="Application\\n(RipeSense UI)" fillcolor="#e3a008" fontcolor="#ffffff"];
      D -> K; D -> M; K -> M; M -> E; M -> A; K -> A;
    }
    """
    st.graphviz_chart(arch, width='stretch')
    st.markdown("### Layer responsibilities")
    for i, (layer, desc, module, artefact) in enumerate(layers):
        st.markdown(
            f"<div class='rulebox' style='--tone:{CARD_TONES[i % len(CARD_TONES)]}'>"
            f"<span class='pill'>{layer}</span> {desc}<br>"
            f"<small><b>Module:</b> <code>{module}</code> &nbsp;·&nbsp; "
            f"<b>Artefact:</b> <code>{artefact}</code></small></div>",
            unsafe_allow_html=True)
    st.caption(
        "Offline benchmark only: no live IoT streams. Decision Support replays the "
        "same inference path on simulated or uploaded sensor readings.")


# --------------------------------------------------------------------------- #
# Page: Overview
# --------------------------------------------------------------------------- #
elif PAGE == "Overview":
    page_header(
        "overview", "RipeSense",
        "Predicting post-harvest banana ripeness from low-cost IoT sensors, "
        "enhanced with a literature-based knowledge graph.",
        eyebrow="MSc Artificial Intelligence Capstone", hero=True)

    best = RESULTS["models"][RESULTS["best_model"]["name"]]["test"]
    c1, c2, c3, c4 = st.columns(4)
    card(c1, "Best macro-F1", f"{best['macro_f1']:.3f}", tone=0)
    card(c2, "Best accuracy", f"{best['accuracy']:.3f}", tone=1)
    card(c3, "KG rules used",
         f"{RESULTS['kg']['n_rules_accepted']}/{RESULTS['kg']['n_rules_total']}",
         tone=2)
    card(c4, "Test samples", f"{RESULTS['data_report']['n_test']:,}", tone=3)

    st.markdown("### How RipeSense works")
    st.markdown(
        "RipeSense fuses **six BME280 sensor readings** with **literature-based "
        "knowledge-graph rules** encoded as transparent tabular features. The KG "
        "supports **interpretability and operator audit** while sensor-only models "
        "already achieve ~99.2% macro-F1 (McNemar: no significant KG accuracy gain). "
        "This is an **offline simulation** of decision support—not live IoT deployment.")

    flow = """
    digraph {
      bgcolor="transparent"; rankdir=LR; pad=0.2; nodesep=0.45; ranksep=0.6;
      node [shape=box style="rounded,filled" fontname="Segoe UI" fontsize=11
            penwidth=1.2 color="#e0d8c8" fontcolor="#241e14" margin="0.22,0.14"];
      edge [color="#c7bfae" penwidth=1.2 arrowsize=0.7];
      A [label="IoT sensors\\n(6 BME280)" fillcolor="#ffffff" color="#cfe0d4"];
      L [label="Post-harvest\\nliterature" fillcolor="#ffffff" color="#e6dcbe"];
      K [label="Knowledge graph\\n(NetworkX)" fillcolor="#eef3e4" color="#cdd9ac"];
      M [label="RF / XGBoost\\n(KG-augmented)" fillcolor="#fdf3d8" color="#eddba6"];
      D [label="Ripeness +\\nstorage advice" fillcolor="#e3a008" fontcolor="#ffffff"
         color="#c98c05"];
      A -> K; L -> K; A -> M; K -> M; M -> D;
    }
    """
    st.graphviz_chart(flow, width='stretch')

    gain_rf = (RESULTS["models"]["kg_rf"]["test"]["macro_f1"]
               - RESULTS["models"]["baseline_rf"]["test"]["macro_f1"])
    gain_xgb = (RESULTS["models"]["kg_xgb"]["test"]["macro_f1"]
                - RESULTS["models"]["baseline_xgb"]["test"]["macro_f1"])
    st.markdown("### Key findings")
    colA, colB = st.columns(2)
    for col, gain, key, label, tone in [
            (colA, gain_rf, "rf_baseline_vs_kg", "RF", BANANA["leaf"]),
            (colB, gain_xgb, "xgb_baseline_vs_kg", "XGBoost", BANANA["amber"])]:
        sig = RESULTS["mcnemar"][key]["significant"]
        col.markdown(
            f"<div class='rulebox' style='--tone:{tone}'><span class='pill'>RQ1</span>"
            f"<b>KG effect ({label})</b><br>macro-F1 change = <b>{gain:+.4f}</b><br>"
            f"McNemar significant: <b>{sig}</b></div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Page: Data Explorer
# --------------------------------------------------------------------------- #
elif PAGE == "Data Explorer":
    page_header("data", "Data Explorer",
                "Training and test partitions, sensor ranges, class balance and the "
                "relationships between the six BME280 channels.")
    dr = RESULTS["data_report"]
    c1, c2, c3 = st.columns(3)
    card(c1, "Train rows", f"{dr['n_train']:,}", tone=0)
    card(c2, "Test rows", f"{dr['n_test']:,}", tone=1)
    card(c3, "Classes balanced?",
         "Yes (20% each)" if not dr["smote_applied"] else "No", tone=2)

    st.subheader("Feature ranges (training set)")
    fs = pd.DataFrame(RESULTS["feature_summary"])
    st.dataframe(fs.style.format("{:.3f}"), width='stretch')

    eda = load_json("eda_summary.json")
    if eda:
        st.subheader("Class distribution")
        dist = dr["class_distribution_train"]
        stages = sorted(int(k) for k in dist.keys())
        fig = px.bar(
            x=[str(s) for s in stages],
            y=[dist[str(s)] for s in stages],
            labels={"x": "Ripeness stage", "y": "Count"},
            color=[str(s) for s in stages],
            color_discrete_map={str(s): STAGE_COLORS[s] for s in stages},
        )
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, width='stretch')

        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("Sensor correlation")
            corr = pd.DataFrame(eda["correlation"])
            fig = px.imshow(corr, text_auto=".2f",
                            color_continuous_scale=["#8c5a2f", "#f4f6f4", ACCENT],
                            zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(height=420, coloraxis_colorbar=dict(
                thickness=10, outlinewidth=0, tickfont=dict(size=10, color=MUTED)))
            st.plotly_chart(fig, width='stretch')
        with cc2:
            st.subheader("Per-stage mean by feature")
            feat = st.selectbox("Feature", C.SENSOR_FEATURES)
            psm = eda["per_stage_mean"][feat]
            fig = px.line(x=list(psm.keys()), y=list(psm.values()), markers=True,
                          labels={"x": "Stage", "y": f"Mean {feat}"})
            fig.update_traces(line_color=ACCENT, marker_color=ACCENT, line_width=2)
            fig.update_layout(height=420)
            st.plotly_chart(fig, width='stretch')

        if fig_path("sensor_by_stage.png"):
            st.subheader("Sensor distributions by stage")
            st.image(fig_path("sensor_by_stage.png"), width='stretch')

    st.subheader("Preprocessing preview (offline replay)")
    st.caption(
        "Dataset: Bath ds_34 · DOI 10.15125/BATH-01459 · previously collected "
        "laboratory recordings, not live IoT streams.")
    absent = missing_artefacts()
    if absent:
        st.info("Load trained models (`py -m src.run_pipeline`) to preview scaling.")
    else:
        scaler, _, _ = load_models()
        preview_vals = {f: float(fs.loc[f, "mean"]) for f in C.SENSOR_FEATURES}
        from src.decision_support import preprocess_preview
        prev = preprocess_preview(scaler, preview_vals)
        st.dataframe(
            pd.DataFrame(prev["rows"]).rename(columns={
                "feature": "Sensor", "raw": "Raw value", "scaled": "Scaled [0–1]",
                "in_range": "In range?", "lo": "Min", "hi": "Max",
            }),
            width='stretch',
            hide_index=True,
        )


# --------------------------------------------------------------------------- #
# Page: Knowledge Graph
# --------------------------------------------------------------------------- #
elif PAGE == "Knowledge Graph":
    page_header("graph", "Knowledge Graph",
                "Post-harvest rules extracted from the literature, each kept or "
                "discarded on statistical evidence from the training data.")
    rules = load_json("validated_rules.json") or []
    accepted = [r for r in rules if r["accepted"]]
    rejected = [r for r in rules if not r["accepted"]]

    c1, c2, c3 = st.columns(3)
    card(c1, "Triples evaluated", f"{len(rules)}", tone=4)
    card(c2, "Accepted", f"{len(accepted)}", tone=0)
    card(c3, "Rejected", f"{len(rejected)}", tone=3)

    st.subheader("Validated rules")
    st.caption("Each rule kept only if activation ≥ 5% AND chi-squared p < 0.05.")
    df = pd.DataFrame(rules)
    show = df[["rule_id", "activation_rate", "p_value", "accepted", "reason",
               "predicate", "object", "expected_dir", "source"]]
    st.dataframe(show.style.format({"activation_rate": "{:.3f}",
                                    "p_value": "{:.3g}"}),
                 width='stretch', height=430)

    st.subheader("Rule activation rates")
    dfa = df.sort_values("activation_rate", ascending=True)
    fig = px.bar(dfa, x="activation_rate", y="rule_id", orientation="h",
                 color="accepted",
                 color_discrete_map={True: ACCENT, False: "#cdbfa4"})
    fig.add_vline(x=C.MIN_ACTIVATION_RATE, line_dash="dot", line_color=STAGE_TEXT[5],
                  line_width=1.4, annotation_text="minimum activation",
                  annotation_font=dict(size=10, color=STAGE_TEXT[5]))
    fig.update_layout(height=460)
    st.plotly_chart(fig, width='stretch')


# --------------------------------------------------------------------------- #
# Page: Model Results
# --------------------------------------------------------------------------- #
elif PAGE == "Model Results":
    page_header("results", "Model Results",
                "Held-out test performance for the sensor-only baselines and their "
                "knowledge-graph-augmented counterparts, with significance testing.")
    rows = []
    for name, r in RESULTS["models"].items():
        rows.append({"model": name, "type": "KG" if r["is_kg"] else "baseline",
                     **r["test"], "cv_macro_f1": r["cv"]["cv_best_macro_f1"]})
    df = pd.DataFrame(rows)
    st.subheader("Test-set performance")
    st.dataframe(df.style.format({c: "{:.4f}" for c in
                 ["accuracy", "macro_f1", "weighted_precision", "weighted_recall", "cv_macro_f1"]}),
                 width='stretch')

    metric = st.selectbox("Compare metric",
                          ["macro_f1", "accuracy", "weighted_precision", "weighted_recall"])
    fig = px.bar(df, x="model", y=metric, color="type", barmode="group",
                 color_discrete_map={"KG": ACCENT, "baseline": "#cdbfa4"},
                 text_auto=".3f")
    fig.update_traces(textposition="outside", textfont=dict(size=11, color=MUTED))
    fig.update_layout(height=420, yaxis_range=[0, 1.08])
    st.plotly_chart(fig, width='stretch')

    st.subheader("RQ1 - Statistical significance (McNemar)")
    mc = RESULTS["mcnemar"]
    cc1, cc2 = st.columns(2)
    for col, key, title in [(cc1, "rf_baseline_vs_kg", "Random Forest"),
                            (cc2, "xgb_baseline_vs_kg", "XGBoost")]:
        v = mc[key]
        col.markdown(f"**{title}: baseline vs KG**")
        col.metric("p-value", f"{v['p_value']:.4g}",
                   "significant" if v["significant"] else "not significant")

    st.subheader("Confusion matrices")
    pick = st.selectbox("Model", list(RESULTS["models"].keys()), index=1)
    p = fig_path(f"confusion_{pick}.png")
    if p:
        st.image(p, width=520)


# --------------------------------------------------------------------------- #
# Page: Interpretability
# --------------------------------------------------------------------------- #
elif PAGE == "Interpretability":
    page_header("interpret", "Interpretability (SHAP)",
                "Which inputs drive each model, and whether the knowledge-graph "
                "features are used in the direction the literature predicts.")
    align = (load_json("rq2_alignment.json") or {}).get("kg_rf", {})
    if align:
        c1, c2, c3 = st.columns(3)
        card(c1, "Rule alignment score", f"{align['alignment_score']:.2f}", tone=3)
        card(c2, "Rules checked", f"{align['checked']}", tone=4)
        card(c3, "Rules aligned", f"{align['aligned']}", tone=0)

    cc1, cc2 = st.columns(2)
    for col, tag, title in [(cc1, "baseline_rf", "Baseline RF"),
                            (cc2, "kg_rf", "KG-augmented RF")]:
        col.subheader(title)
        imp = load_json(f"shap_importance_{tag}.json")
        if imp:
            col.caption(f"Method: {imp['method']}")
            s = pd.Series(imp["importance"]).sort_values(ascending=True).tail(15)
            fig = px.bar(x=s.values, y=s.index, orientation="h",
                         color_discrete_sequence=[ACCENT])
            fig.update_layout(height=460, xaxis_title="Importance", yaxis_title="")
            col.plotly_chart(fig, width='stretch')

    if align and align.get("details"):
        st.subheader("RQ2 - Rule-direction alignment detail")
        st.dataframe(pd.DataFrame(align["details"]), width='stretch')


# --------------------------------------------------------------------------- #
# Page: Robustness
# --------------------------------------------------------------------------- #
elif PAGE == "Robustness":
    page_header("robust", "Robustness (RQ3)",
                "Macro-F1 retained as a percentage of clean-data performance under "
                "sensor noise, missing values, and dual sensor failure.")
    rob = RESULTS["robustness"]

    for kind, title in [("noise", "Gaussian noise"), ("missing", "Missing values")]:
        st.subheader(title)
        fig = go.Figure()
        for model_name, res in rob.items():
            levels = list(res[kind].keys())
            pcts = [res[kind][l]["pct_of_clean"] for l in levels]
            is_kg = model_name.startswith("kg")
            colour = ACCENT if is_kg else "#bfae90"
            fig.add_trace(go.Scatter(
                x=levels, y=pcts, mode="lines+markers", name=model_name,
                line=dict(width=2.2, color=colour,
                          dash="solid" if is_kg else "dash"),
                marker=dict(size=7, color=colour,
                            symbol="circle" if is_kg else "diamond")))
        fig.add_hline(y=80, line_dash="dot", line_color=STAGE_TEXT[5], line_width=1.4,
                      annotation_text="80% threshold",
                      annotation_font=dict(size=10, color=STAGE_TEXT[5]))
        fig.update_layout(height=360, yaxis_title="Macro-F1 (% of clean)",
                          yaxis_range=[0, 105], xaxis_title="Degradation level",
                          xaxis_type="category")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Dual sensor failure (Temp-int + Temp-ext)")
    cc = st.columns(len(rob))
    for i, (col, (name, res)) in enumerate(zip(cc, rob.items())):
        sf = res["sensor_failure"]
        card(col, f"{name}", f"{sf['pct_of_clean']}%", tone=i)


# --------------------------------------------------------------------------- #
# Page: Decision Support (compare baseline vs KG)
# --------------------------------------------------------------------------- #
elif PAGE == "Decision Support":
    page_header("live", "Decision Support",
                "Simulate or upload six BME280 readings. Compare sensor-only and "
                "knowledge-augmented predictions side-by-side, inspect KG features, "
                "fired rules, and local explanations. Advisory only—not Brix, "
                "shelf-life, or live IoT deployment.")

    absent = missing_artefacts()
    if absent:
        st.warning("**Trained model files are not present.** Run the pipeline first:")
        st.code("py -m src.run_pipeline", language="bash")
        st.caption("Missing: " + ", ".join(absent))
        st.stop()

    scaler, gen, models = load_models()
    fs = pd.DataFrame(RESULTS["feature_summary"])
    defaults = {f: float(fs.loc[f, "mean"]) for f in C.SENSOR_FEATURES}

    input_mode = st.radio(
        "Input mode",
        ["Manual sliders", "Pick test-set row", "Upload CSV (one row)"],
        horizontal=True,
        key="ds_input_mode",
    )

    vals = dict(defaults)
    ground_truth = None

    if input_mode == "Pick test-set row":
        samples = load_test_samples()
        if samples.empty:
            st.error("Test CSV not found in data/ds_34/.")
            st.stop()
        options = [
            f"Row {r.row_id} · actual stage {int(r[C.LABEL_NAME])}"
            for r in samples.itertuples()
        ]
        pick = st.selectbox("Test sample (ground truth for demo only)", options)
        idx = options.index(pick)
        row = samples.iloc[idx]
        for f in C.SENSOR_FEATURES:
            vals[f] = float(row[f])
        ground_truth = int(row[C.LABEL_NAME])
        st.info(f"Loaded test row — labelled stage **{ground_truth}** (offline benchmark).")

    elif input_mode == "Upload CSV (one row)":
        up = st.file_uploader("CSV with columns: " + ", ".join(C.SENSOR_FEATURES),
                              type=["csv"])
        if up is not None:
            df_up = pd.read_csv(up)
            miss = [c for c in C.SENSOR_FEATURES if c not in df_up.columns]
            if miss:
                st.error("Missing columns: " + ", ".join(miss))
            else:
                row = df_up.iloc[0]
                for f in C.SENSOR_FEATURES:
                    vals[f] = float(row[f])
                st.success("CSV row loaded.")

    algo = st.radio("Algorithm", ["XGBoost", "Random Forest"], horizontal=True,
                    key="ds_algo")
    algo_key = "xgb" if algo.startswith("XGB") else "rf"

    if input_mode == "Manual sliders":
        groups = [
            ("enclosure", "Internal sensors", ["Temp-int", "Humid-int", "Press-int"]),
            ("ambient", "Ambient sensors", ["Temp-ext", "Humid-ext", "Press-ext"]),
        ]
        gcols = st.columns(2)
        for gcol, (glyph, title, feats) in zip(gcols, groups):
            with gcol, st.container(border=True):
                st.markdown(f"<div class='sensor-group-title'>{icon(glyph, 16)}"
                            f"<span>{title}</span></div>", unsafe_allow_html=True)
                for feat in feats:
                    lo = float(fs.loc[feat, "min"])
                    hi = float(fs.loc[feat, "max"])
                    mean = float(fs.loc[feat, "mean"])
                    name, unit, _ = FEATURE_META[feat]
                    vals[feat] = st.slider(
                        f"{name} ({unit})", lo, hi, mean,
                        step=(hi - lo) / 200 or 0.1,
                        key=f"ds_{feat.replace('-', '_')}",
                    )

    compare = predict_compare(models, scaler, gen, vals, algorithm=algo_key)

    if compare["disagree"]:
        st.warning(
            f"Models **disagree**: baseline stage {compare['baseline']['predicted_stage']} "
            f"vs KG stage {compare['kg']['predicted_stage']} — consistent with sparse "
            "McNemar discordant pairs on the full test set.")
    else:
        st.success(
            f"Models **agree** on stage {compare['baseline']['predicted_stage']} "
            f"(confidence baseline {compare['baseline']['confidence']*100:.1f}% · "
            f"KG {compare['kg']['confidence']*100:.1f}%).")

    c1, c2 = st.columns(2)
    for col, side, key in [(c1, "Sensor-only baseline", "baseline"),
                           (c2, "KG-augmented", "kg")]:
        pred = compare[key]
        sc = STAGE_COLORS[pred["predicted_stage"]]
        col.markdown(
            f"<div class='stage-banner' style='--sc:{sc};margin-bottom:0.8rem'>"
            f"<span class='sw'><i></i></span><div>"
            f"<div class='t'>{side}</div>"
            f"<div class='t'>Stage {pred['predicted_stage']} — "
            f"{C.STAGE_LABELS[pred['predicted_stage']].split(' - ')[1]}</div>"
            f"<small>Confidence {pred['confidence']*100:.1f}%</small></div></div>",
            unsafe_allow_html=True)
        probs = pred["class_probabilities"]
        fig = px.bar(
            x=[f"S{s}" for s in sorted(probs.keys())],
            y=[probs[s] for s in sorted(probs.keys())],
            labels={"x": "Stage", "y": "Probability"},
            color=[f"S{s}" for s in sorted(probs.keys())],
            color_discrete_map={
                f"S{s}": STAGE_COLORS[s] for s in sorted(probs.keys())
            },
        )
        fig.update_layout(showlegend=False, height=260, yaxis_range=[0, 1.05])
        col.plotly_chart(fig, use_container_width=True)

    st.subheader("Knowledge-graph features (generated for this reading)")
    st.dataframe(compare["kg_features"].T.rename(columns={0: "value"}),
                 width='stretch')

    st.subheader("Preprocessing trace")
    st.dataframe(pd.DataFrame(compare["preprocess"]["rows"]), width='stretch',
                 hide_index=True)

    st.subheader("Local feature importance (this model)")
    lc1, lc2 = st.columns(2)
    lc1.markdown("**Baseline**")
    lc1.dataframe(pd.DataFrame(compare["baseline_local_features"]),
                  hide_index=True, width='stretch')
    lc2.markdown("**KG-augmented**")
    lc2.dataframe(pd.DataFrame(compare["kg_local_features"]),
                  hide_index=True, width='stretch')
    st.caption(
        "Local panel uses tree impurity importances for the fitted model; global "
        "batch SHAP plots are on the Interpretability page.")

    st.subheader("Fired knowledge-graph rules & storage advice")
    for i, rec in enumerate(compare["kg"]["recommendations"]):
        st.markdown(f"<div class='rulebox' style='animation-delay:{0.05*i:.2f}s'>"
                    f"{rec}</div>", unsafe_allow_html=True)
    if compare["kg"]["fired_rules"]:
        for i, r in enumerate(compare["kg"]["fired_rules"]):
            st.markdown(
                f"<div class='firedrule' style='animation-delay:{0.05*i:.2f}s'>"
                f"<span class='pill'>{r['rule_id']}</span> {r['text']} "
                f"<em style='color:{MUTED}'>{r['source']}</em></div>",
                unsafe_allow_html=True)
    else:
        st.info("No rules fired — readings within normal mid-range bounds.")

st.markdown("<div style='margin-top:2.6rem;padding-top:1rem;"
            "border-top:1px solid var(--line);color:#8b9891;font-size:0.78rem;"
            "line-height:1.5'>RipeSense &nbsp;·&nbsp; MSc Artificial Intelligence "
            "Capstone Project &nbsp;·&nbsp; Knowledge-Integrated Supervised Learning "
            "for Post-Harvest Banana Ripeness Prediction</div>",
            unsafe_allow_html=True)
