"""RipeSense - Knowledge-Integrated Banana Ripeness Decision-Support App.

An interactive Streamlit dashboard that visualises the data, the knowledge
graph, model performance, interpretability and robustness, and provides a live
prediction tool. Run with:  py -m streamlit run app.py
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.decision_support import predict_one
from src.kg_features import KGFeatureGenerator

# --------------------------------------------------------------------------- #
# Page config & theme
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="RipeSense - Banana Ripeness Decision Support",
                   page_icon="🍌", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
    .stApp { background: linear-gradient(160deg, #fffef7 0%, #f4f9ef 100%); }
    .hero {
        background: linear-gradient(120deg, #f9d423 0%, #a8e063 60%, #56ab2f 100%);
        padding: 2rem 2.4rem; border-radius: 20px; color: #1b3a0e;
        box-shadow: 0 10px 30px rgba(86,171,47,0.25); margin-bottom: 1.4rem;
    }
    .hero h1 { font-size: 2.5rem; margin: 0; font-weight: 800; letter-spacing: -1px; }
    .hero p { font-size: 1.05rem; margin: 0.4rem 0 0 0; opacity: 0.85; }
    .metric-card {
        background: #ffffff; border-radius: 16px; padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06); border-left: 6px solid #56ab2f;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; color: #2c5f17; }
    .metric-card span { color: #6b7280; font-size: 0.85rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.5px; }
    .pill { display:inline-block; background:#eef7e6; color:#3c6b1e;
        padding:0.25rem 0.7rem; border-radius:999px; font-size:0.8rem;
        margin:0.15rem; font-weight:600; }
    .rulebox { background:#ffffff; border-radius:12px; padding:0.8rem 1rem;
        border-left:4px solid #f9b115; margin-bottom:0.5rem; color:#1f2937 !important;
        box-shadow:0 2px 8px rgba(0,0,0,0.04); font-size:0.95rem; }
    .rulebox b, .rulebox strong { color:#2c5f17; }
    .stage-banner { padding:1.4rem; border-radius:16px; text-align:center;
        font-size:1.4rem; font-weight:800; color:#1b3a0e !important; }
    .stage-banner small { display:block; font-size:0.95rem; font-weight:600;
        margin-top:0.3rem; color:#2c5f17 !important; opacity:1; }
    .sensor-group { background:#ffffff; border-radius:14px; padding:0.8rem 1rem 1rem;
        box-shadow:0 3px 14px rgba(0,0,0,0.05); margin-bottom:0.6rem;
        border:1px solid #e8f5e9; }
    .sensor-group-title { color:#2c5f17; font-weight:700; font-size:1.05rem;
        margin:0 0 0.6rem 0; padding-bottom:0.35rem; border-bottom:2px solid #c5e1a5; }
    .firedrule { background:#ffffff; border-radius:12px; padding:0.7rem 1rem;
        border-left:4px solid #f9b115; margin-bottom:0.45rem; color:#1f2937;
        box-shadow:0 2px 8px rgba(0,0,0,0.04); }
    /* Force all widget labels / questions in the main area to be readable */
    section[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stMain"] [data-testid="stWidgetLabel"] label,
    section[data-testid="stMain"] label p {
        color:#1b3a0e !important; font-weight:600 !important; font-size:0.95rem; }
    section[data-testid="stMain"] [data-testid="stRadio"] label p,
    section[data-testid="stMain"] [role="radiogroup"] label {
        color:#1b3a0e !important; font-weight:600 !important; }
    section[data-testid="stMain"] [data-testid="stExpander"] summary p,
    section[data-testid="stMain"] details summary {
        color:#1b3a0e !important; font-weight:700 !important; }
    section[data-testid="stMain"] .stCheckbox label p { color:#1b3a0e !important; }
    /* Headings, captions and body text in main content */
    section[data-testid="stMain"] h1,
    section[data-testid="stMain"] h2,
    section[data-testid="stMain"] h3,
    section[data-testid="stMain"] h4,
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stMain"] [data-testid="stCaptionContainer"] p,
    section[data-testid="stMain"] [data-testid="stHeader"] {
        color:#1b3a0e !important; }
    section[data-testid="stMain"] [data-testid="stCaptionContainer"] p {
        color:#374151 !important; font-weight:500; }
    /* Slider thumb track labels remain readable */
    section[data-testid="stMain"] [data-testid="stSlider"] [data-testid="stWidgetLabel"] p {
        color:#1b3a0e !important; font-weight:700 !important; }
    /* Sensor reading chips (input echo) */
    .sensor-chip { background:#ffffff; border-radius:14px; padding:0.7rem 0.6rem;
        box-shadow:0 3px 12px rgba(0,0,0,0.07); border-top:4px solid #56ab2f;
        text-align:center; margin-bottom:0.5rem; }
    .sensor-chip .ic { font-size:1.6rem; line-height:1; }
    .sensor-chip .nm { color:#33691e; font-weight:700; font-size:0.78rem;
        margin-top:0.25rem; line-height:1.1; }
    .sensor-chip .vl { color:#1b3a0e; font-weight:800; font-size:1.15rem;
        margin-top:0.2rem; }
    section[data-testid="stSidebar"] { background: #1b3a0e; }
    section[data-testid="stSidebar"] * { color: #eaf6e1 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

STAGE_COLORS = {1: "#2e7d32", 2: "#9ccc65", 3: "#cddc39", 4: "#ffca28", 5: "#a1632e"}
# Darker, high-contrast variants for TEXT on a white/light background.
STAGE_TEXT = {1: "#1b5e20", 2: "#558b2f", 3: "#827717", 4: "#b26a00", 5: "#5d4037"}
STAGE_EMOJI = {1: "🟢", 2: "🟢🟡", 3: "🟡", 4: "🍌", 5: "🟤"}

# Friendly label, unit and icon for each sensor feature.
FEATURE_META = {
    "Temp-int": ("Internal Temperature", "°C", "🌡️"),
    "Humid-int": ("Internal Humidity", "%RH", "💧"),
    "Press-int": ("Internal Pressure", "hPa", "🫧"),
    "Temp-ext": ("Ambient Temperature", "°C", "🌡️"),
    "Humid-ext": ("Ambient Humidity", "%RH", "💧"),
    "Press-ext": ("Ambient Pressure", "hPa", "🫧"),
}


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


def card(col, label, value):
    col.markdown(
        f"<div class='metric-card'><span>{label}</span><h2>{value}</h2></div>",
        unsafe_allow_html=True)


RESULTS = load_results()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.markdown("## 🍌 RipeSense")
st.sidebar.caption("Knowledge-Integrated Ripeness Monitoring")
PAGE = st.sidebar.radio(
    "Navigate",
    ["Overview", "Data Explorer", "Knowledge Graph", "Model Results",
     "Interpretability", "Robustness", "Live Prediction"],
    key="nav_page",
)
st.sidebar.markdown("---")
if RESULTS:
    bm = RESULTS["best_model"]["name"]
    st.sidebar.success(f"Pipeline loaded ✓\nBest model: {bm}")
else:
    st.sidebar.error("Run the pipeline first:\n`py -m src.run_pipeline`")

if RESULTS is None:
    st.markdown("<div class='hero'><h1>🍌 RipeSense</h1>"
                "<p>No results found yet. Please run "
                "<code>py -m src.run_pipeline</code> to generate outputs.</p></div>",
                unsafe_allow_html=True)
    st.stop()


# --------------------------------------------------------------------------- #
# Page: Overview
# --------------------------------------------------------------------------- #
if PAGE == "Overview":
    st.markdown(
        "<div class='hero'><h1>🍌 RipeSense</h1>"
        "<p>Predicting post-harvest banana ripeness from low-cost IoT sensors, "
        "enhanced with a literature-based knowledge graph.</p></div>",
        unsafe_allow_html=True)

    best = RESULTS["models"][RESULTS["best_model"]["name"]]["test"]
    c1, c2, c3, c4 = st.columns(4)
    card(c1, "Best macro-F1", f"{best['macro_f1']:.3f}")
    card(c2, "Best accuracy", f"{best['accuracy']:.3f}")
    card(c3, "KG rules used", f"{RESULTS['kg']['n_rules_accepted']}/{RESULTS['kg']['n_rules_total']}")
    card(c4, "Test samples", f"{RESULTS['data_report']['n_test']:,}")

    st.markdown("### How RipeSense works")
    st.markdown(
        "RipeSense fuses **six BME280 sensor readings** with **expert post-harvest "
        "knowledge** encoded as a knowledge graph. Validated rules become extra "
        "model features, improving accuracy, interpretability and robustness.")

    flow = """
    digraph {
      rankdir=LR; node [shape=box style="rounded,filled" fontname=Helvetica];
      A [label="IoT sensors\\n(6 BME280)" fillcolor="#d4e157"];
      L [label="Post-harvest\\nliterature" fillcolor="#ffe082"];
      K [label="Knowledge Graph\\n(NetworkX)" fillcolor="#a8e063"];
      M [label="RF / XGBoost\\n(KG-augmented)" fillcolor="#81c784"];
      D [label="Ripeness +\\nstorage advice" fillcolor="#4caf50" fontcolor=white];
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
    colA.info(f"**RQ1 - KG effect (RF):** macro-F1 change = **{gain_rf:+.4f}**  \n"
              f"McNemar significant: **{RESULTS['mcnemar']['rf_baseline_vs_kg']['significant']}**")
    colB.info(f"**RQ1 - KG effect (XGBoost):** macro-F1 change = **{gain_xgb:+.4f}**  \n"
              f"McNemar significant: **{RESULTS['mcnemar']['xgb_baseline_vs_kg']['significant']}**")


# --------------------------------------------------------------------------- #
# Page: Data Explorer
# --------------------------------------------------------------------------- #
elif PAGE == "Data Explorer":
    st.header("📊 Data Explorer")
    dr = RESULTS["data_report"]
    c1, c2, c3 = st.columns(3)
    card(c1, "Train rows", f"{dr['n_train']:,}")
    card(c2, "Test rows", f"{dr['n_test']:,}")
    card(c3, "Classes balanced?", "Yes (20% each)" if not dr["smote_applied"] else "No")

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
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(height=420)
            st.plotly_chart(fig, width='stretch')
        with cc2:
            st.subheader("Per-stage mean by feature")
            feat = st.selectbox("Feature", C.SENSOR_FEATURES)
            psm = eda["per_stage_mean"][feat]
            fig = px.line(x=list(psm.keys()), y=list(psm.values()), markers=True,
                          labels={"x": "Stage", "y": f"Mean {feat}"})
            fig.update_traces(line_color="#56ab2f")
            fig.update_layout(height=420)
            st.plotly_chart(fig, width='stretch')

        if fig_path("sensor_by_stage.png"):
            st.subheader("Sensor distributions by stage")
            st.image(fig_path("sensor_by_stage.png"), width='stretch')


# --------------------------------------------------------------------------- #
# Page: Knowledge Graph
# --------------------------------------------------------------------------- #
elif PAGE == "Knowledge Graph":
    st.header("🕸️ Knowledge Graph")
    rules = load_json("validated_rules.json") or []
    accepted = [r for r in rules if r["accepted"]]
    rejected = [r for r in rules if not r["accepted"]]

    c1, c2, c3 = st.columns(3)
    card(c1, "Triples evaluated", f"{len(rules)}")
    card(c2, "Accepted", f"{len(accepted)}")
    card(c3, "Rejected", f"{len(rejected)}")

    st.subheader("Validated rules")
    st.caption("Each rule kept only if activation ≥ 5% AND chi-squared p < 0.05.")
    df = pd.DataFrame(rules)
    show = df[["rule_id", "activation_rate", "p_value", "accepted", "reason",
               "predicate", "object", "expected_dir", "source"]]
    st.dataframe(show, width='stretch', height=430)

    st.subheader("Rule activation rates")
    dfa = df.sort_values("activation_rate", ascending=True)
    fig = px.bar(dfa, x="activation_rate", y="rule_id", orientation="h",
                 color="accepted", color_discrete_map={True: "#56ab2f", False: "#e57373"})
    fig.add_vline(x=C.MIN_ACTIVATION_RATE, line_dash="dash", line_color="black")
    fig.update_layout(height=460)
    st.plotly_chart(fig, width='stretch')


# --------------------------------------------------------------------------- #
# Page: Model Results
# --------------------------------------------------------------------------- #
elif PAGE == "Model Results":
    st.header("🎯 Model Results")
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
                 color_discrete_map={"KG": "#56ab2f", "baseline": "#bdbdbd"}, text_auto=".3f")
    fig.update_layout(height=420, yaxis_range=[0, 1.05])
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
    st.header("🔍 Interpretability (SHAP)")
    align = (load_json("rq2_alignment.json") or {}).get("kg_rf", {})
    if align:
        c1, c2, c3 = st.columns(3)
        card(c1, "Rule alignment score", f"{align['alignment_score']:.2f}")
        card(c2, "Rules checked", f"{align['checked']}")
        card(c3, "Rules aligned", f"{align['aligned']}")

    cc1, cc2 = st.columns(2)
    for col, tag, title in [(cc1, "baseline_rf", "Baseline RF"),
                            (cc2, "kg_rf", "KG-augmented RF")]:
        col.subheader(title)
        imp = load_json(f"shap_importance_{tag}.json")
        if imp:
            col.caption(f"Method: {imp['method']}")
            s = pd.Series(imp["importance"]).sort_values(ascending=True).tail(15)
            fig = px.bar(x=s.values, y=s.index, orientation="h",
                         color_discrete_sequence=["#56ab2f"])
            fig.update_layout(height=460, xaxis_title="Importance", yaxis_title="")
            col.plotly_chart(fig, width='stretch')

    if align and align.get("details"):
        st.subheader("RQ2 - Rule-direction alignment detail")
        st.dataframe(pd.DataFrame(align["details"]), width='stretch')


# --------------------------------------------------------------------------- #
# Page: Robustness
# --------------------------------------------------------------------------- #
elif PAGE == "Robustness":
    st.header("🛡️ Robustness (RQ3)")
    rob = RESULTS["robustness"]
    st.caption("Macro-F1 retained as a percentage of clean-data performance under "
               "sensor noise, missing values, and dual sensor failure.")

    for kind, title in [("noise", "Gaussian noise"), ("missing", "Missing values")]:
        st.subheader(title)
        fig = go.Figure()
        for model_name, res in rob.items():
            levels = list(res[kind].keys())
            pcts = [res[kind][l]["pct_of_clean"] for l in levels]
            fig.add_trace(go.Scatter(x=levels, y=pcts, mode="lines+markers",
                                     name=model_name))
        fig.add_hline(y=80, line_dash="dash", line_color="red",
                      annotation_text="80% threshold")
        fig.update_layout(height=360, yaxis_title="Macro-F1 (% of clean)",
                          yaxis_range=[0, 105], xaxis_title="Degradation level")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Dual sensor failure (Temp-int + Temp-ext)")
    cc = st.columns(len(rob))
    for col, (name, res) in zip(cc, rob.items()):
        sf = res["sensor_failure"]
        card(col, f"{name}", f"{sf['pct_of_clean']}%")


# --------------------------------------------------------------------------- #
# Page: Live Prediction
# --------------------------------------------------------------------------- #
elif PAGE == "Live Prediction":
    st.header("⚡ Live Ripeness Prediction")
    st.caption("Set the six IoT sensor readings below. RipeSense predicts the "
               "banana ripeness stage, shows which knowledge-graph rules fired, "
               "and gives storage advice.")

    scaler, gen, models = load_models()
    fs = pd.DataFrame(RESULTS["feature_summary"])

    # Quick reference of what each ripeness stage means.
    with st.expander("ℹ️ What do the ripeness stages mean?", expanded=False):
        ref = st.columns(5)
        for s in C.RIPENESS_STAGES:
            ref[s - 1].markdown(
                f"<div style='text-align:center'><div style='font-size:1.6rem'>"
                f"{STAGE_EMOJI[s]}</div><b style='color:{STAGE_TEXT[s]}'>Stage {s}</b>"
                f"<div style='font-size:0.8rem;color:#444'>{C.STAGE_LABELS[s].split(' - ')[1]}</div></div>",
                unsafe_allow_html=True)

    with st.form("predict"):
        vals = {}
        groups = [("📍 Internal sensors (inside the fruit enclosure)",
                   ["Temp-int", "Humid-int", "Press-int"]),
                  ("🌤️ Ambient sensors (surrounding environment)",
                   ["Temp-ext", "Humid-ext", "Press-ext"])]
        gcols = st.columns(2)
        for gcol, (title, feats) in zip(gcols, groups):
            with gcol:
                st.markdown(f"<div class='sensor-group'>"
                            f"<div class='sensor-group-title'>{title}</div>",
                            unsafe_allow_html=True)
                for feat in feats:
                    lo = float(fs.loc[feat, "min"])
                    hi = float(fs.loc[feat, "max"])
                    mean = float(fs.loc[feat, "mean"])
                    name, unit, icon = FEATURE_META[feat]
                    vals[feat] = st.slider(
                        f"{icon} {name} ({unit})",
                        lo, hi, mean, step=(hi - lo) / 200 or 0.1,
                        help=f"Observed range in training data: "
                             f"{lo:.1f} - {hi:.1f} {unit}",
                        key=f"input_{feat.replace('-', '_')}",
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        model_choice = st.radio(
            "Model", ["KG-augmented (recommended)", "Sensor-only baseline"],
            horizontal=True, key="model_choice")
        submitted = st.form_submit_button("🍌 Predict ripeness", type="primary",
                                          key="predict_btn")

    if submitted:
        use_kg = model_choice.startswith("KG")
        model = models["kg_rf"] if use_kg else models["baseline_rf"]
        out = predict_one(model, scaler, gen, vals, is_kg=use_kg)
        stage = out["predicted_stage"]
        color = STAGE_COLORS[stage]
        desc = C.STAGE_LABELS[stage].split(" - ")[1]

        # Pick readable text colour for the banner depending on background tint.
        st.markdown(
            f"<div class='stage-banner' style='background:{color}26;"
            f"border:2px solid {color};color:#1b3a0e'>"
            f"{STAGE_EMOJI[stage]} Predicted: Stage {stage} &mdash; {desc}"
            f"<small>Confidence {out['confidence']*100:.1f}% · "
            f"model: {'KG-augmented' if use_kg else 'sensor-only baseline'}</small></div>",
            unsafe_allow_html=True)

        # Echo back exactly what was entered, with icon, name, value and unit.
        st.markdown("#### 🧾 Your input readings")
        in_cols = st.columns(6)
        for ic, feat in zip(in_cols, C.SENSOR_FEATURES):
            name, unit, icon = FEATURE_META[feat]
            ic.markdown(
                f"<div class='sensor-chip'><div class='ic'>{icon}</div>"
                f"<div class='nm'>{name}</div>"
                f"<div class='vl'>{vals[feat]:.1f}<span style='font-size:0.8rem'> {unit}</span></div>"
                f"</div>",
                unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1.2])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=out["confidence"] * 100,
                number={"suffix": "%", "font": {"color": "#1b3a0e"}},
                title={"text": "Prediction confidence", "font": {"size": 16}},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": color},
                       "steps": [{"range": [0, 50], "color": "#fdecea"},
                                 {"range": [50, 80], "color": "#fff8e1"},
                                 {"range": [80, 100], "color": "#e8f5e9"}]}))
            fig.update_layout(height=320, margin=dict(t=60, b=10))
            st.plotly_chart(fig, width='stretch')
        with c2:
            probs = out["class_probabilities"]
            stages_sorted = sorted(probs.keys(), key=lambda x: int(x))
            x_labels = [f"Stage {s}" for s in stages_sorted]
            y_vals = [probs[s] for s in stages_sorted]
            fig = px.bar(
                x=x_labels, y=y_vals,
                color=x_labels,
                color_discrete_map={f"Stage {s}": STAGE_COLORS[int(s)] for s in stages_sorted},
                text=[f"{probs[s]*100:.1f}%" for s in stages_sorted],
                labels={"x": "Ripeness stage", "y": "Probability"})
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, height=320,
                              yaxis_range=[0, 1.12], yaxis_tickformat=".0%",
                              title="Probability per ripeness stage",
                              margin=dict(t=60, b=10))
            st.plotly_chart(fig, width='stretch')

        st.subheader("🔔 Storage recommendations")
        for rec in out["recommendations"]:
            st.markdown(f"<div class='rulebox'>{rec}</div>", unsafe_allow_html=True)

        st.subheader("⚙️ Knowledge-graph rules that fired")
        if out["fired_rules"]:
            for r in out["fired_rules"]:
                st.markdown(
                    f"<div class='firedrule'><span class='pill'>{r['rule_id']}</span> "
                    f"{r['text']} &nbsp;<em style='color:#888'>{r['source']}</em></div>",
                    unsafe_allow_html=True)
        else:
            st.info("No knowledge-graph rules fired for these readings — the "
                    "values sit within normal mid-range bounds.")

st.markdown("---")
st.caption("RipeSense · MSc Artificial Intelligence Capstone Project · "
           "Knowledge-Integrated Supervised Learning for Post-Harvest Banana Ripeness Prediction")
