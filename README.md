# RipeSense — Knowledge-Integrated Banana Ripeness Prediction

**Author:** Arbind Kumar Gauro (A00074251)  
**Supervisor:** Mohammad Javaheri  
**Programme:** MSc Artificial Intelligence · University of Roehampton  

Reproducible software artefact for **offline** post-harvest banana ripeness **stage classification** (1–5) from six BME280 environmental sensors on the Bath `ds_34` benchmark, augmented with literature-based knowledge-graph features (binary flags, risk scores, violation counts).

**Dataset:** [10.15125/BATH-01459](https://doi.org/10.15125/BATH-01459)

---

## Quick start

Requires **Python 3.11+**. From this directory:

```bash
py -m pip install -r requirements.txt
py run_app.py
```

Open **http://localhost:8501**. Trained models are committed under `outputs/models/`, so the dashboard runs immediately after install.

| Goal | Command | Time |
|------|---------|------|
| Launch decision-support app | `py run_app.py` | ~30 s |
| Reproduce all experiments | `py -m src.run_pipeline` | ~4–5 min |
| Verify resubmission features | `py scripts/verify_resubmission.py` | ~5 s |

On macOS/Linux, use `python3` instead of `py`.

---

## What this system does

1. Loads and preprocesses Bath `ds_34` (18,819 train / 8,066 test rows, six sensors).  
2. Validates 15 literature KG rules → **13 accepted** → 32 tabular features.  
3. Trains **four models** (Random Forest & XGBoost × baseline vs KG).  
4. Evaluates RQ1 (McNemar), RQ2 (SHAP alignment), RQ3 (noise / missing / failure).  
5. Serves predictions through a **Streamlit** app with architecture view and **side-by-side baseline vs KG comparison**.

The KG supports **interpretability and auditability**; McNemar tests show **no statistically significant accuracy gain** over sensor-only baselines on this dataset (~99.2% macro-F1). This is **not** live IoT deployment, Brix/shelf-life forecasting, or autonomous ripening control.

---

## Application pages

| Page | Purpose |
|------|---------|
| **System Architecture** | Five-layer design; modules and artefact files |
| **Overview** | Summary metrics and RQ1 findings |
| **Data Explorer** | Dataset stats, correlation, preprocessing preview |
| **Knowledge Graph** | Validated / rejected rules |
| **Model Results** | Four-model comparison, McNemar, confusion matrices |
| **Interpretability** | Global SHAP plots, alignment score |
| **Robustness** | Degradation curves (RQ3) |
| **Decision Support** | Compare baseline vs KG; KG feature table; CSV / test-row input |

---

## Key results (held-out test)

| Metric | Value |
|--------|-------|
| Best model | `kg_xgb` |
| Macro-F1 / accuracy | **0.9932** |
| KG rules accepted | 13 / 15 |
| McNemar (XGB baseline vs KG) | p = 0.522 (not significant) |
| SHAP rule alignment | 0.33 |
| Noise 5% | Below 80% of clean F1 |

Full metrics: `outputs/results/model_results.json`

---

## Project structure

```
RipeSense/
├── run_app.py              # Entry point — install deps + launch Streamlit
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── src/                    # Pipeline modules
│   ├── run_pipeline.py     # Phases 1–6 orchestration
│   ├── data_loader.py
│   ├── kg_features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── robustness.py
│   └── decision_support.py
├── data/ds_34/             # Bath train/test CSVs
├── data/kg/                # Knowledge-graph nodes & edges
├── outputs/
│   ├── figures/            # Evaluation PNGs (for report & app)
│   ├── results/            # JSON metrics
│   └── models/             # Trained estimators + scaler + KG generator
├── docs/screenshots/       # App screenshots
└── scripts/
    ├── verify_setup.py
    └── verify_resubmission.py
```

---

## Pipeline phases

| Phase | Module | Output |
|-------|--------|--------|
| 1 | `data_loader.py` | Scaled splits, `scaler.pkl` |
| 2 | `eda.py` | EDA figures, quartiles |
| 3 | `knowledge_graph.py`, `kg_features.py` | `kg_generator.json`, validated rules |
| 4 | `train.py` | Four `.pkl` models |
| 5 | `evaluate.py`, `robustness.py` | Metrics, SHAP, robustness JSON & figures |
| 6 | `app.py` | Interactive decision support |

---

## Citation

> Callaghan, K. & Martinez Hernandez, U. (2025). *Dataset for Low-Cost, Multi-Sensor Non-Destructive Banana Ripeness Estimation Using Machine Learning.* University of Bath Research Data Archive. https://doi.org/10.15125/BATH-01459

---

## Licence

MIT — see [LICENSE](LICENSE).
