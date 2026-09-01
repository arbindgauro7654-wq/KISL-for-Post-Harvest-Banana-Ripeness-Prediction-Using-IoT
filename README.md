# RipeSense — Knowledge-Integrated Banana Ripeness Prediction

**Author:** Arbind Kumar Gauro (A00074251)  
**Supervisor:** Mohammad Javaheri  
**Programme:** MSc Artificial Intelligence  
**Institution:** University of Roehampton  

---

## Start here — how to run this project

Only **two files** are meant to be run directly. Everything inside `src/` is a
library module imported by the pipeline, not something to open individually.

| I want to… | Run this file | Command | Time |
|------------|---------------|---------|------|
| **See the dashboard / live demo** | `run_app.py` | `py run_app.py` | ~30 s |
| **Reproduce all experiments and results** | `src/run_pipeline.py` | `py -m src.run_pipeline` | ~4–5 min |

### The three-step version

Requires **Python 3.11+** (tested on 3.14, Windows). From inside the `RipeSense/` folder:

```bash
# 1. Install dependencies
py -m pip install -r requirements.txt

# 2. Launch the dashboard  (models are already committed, so this works immediately)
py run_app.py

# 3. Optional — regenerate every result from scratch
py -m src.run_pipeline
```

`run_app.py` checks the Python version, installs anything missing, trains the
models if they are absent, and then opens the app. When it finishes it prints a
URL — open **http://localhost:8501** in a browser and use the left sidebar to
move between the eight pages. **Decision Support** provides side-by-side baseline vs KG comparison.

Press `Ctrl+C` in the terminal to stop the dashboard.

On macOS or Linux, replace `py` with `python3` in every command.

---

## Overview

RipeSense is a reproducible software artefact for **offline post-harvest banana ripeness stage classification** using six low-cost **BME280** sensor readings on the Bath `ds_34` benchmark, augmented with **literature-based knowledge-graph features** (binary flags, risk scores, violation counts). The KG supports **interpretability and auditability**; McNemar tests show **no statistically significant accuracy gain** over sensor-only baselines on this dataset (~99.2% macro-F1). The Streamlit app integrates data management, KG feature generation, model inference, and side-by-side comparison—not live IoT deployment.

```mermaid
flowchart LR
    subgraph inputs [Inputs]
        DS["Bath ds_34 dataset"]
        LIT["Post-harvest literature"]
    end
    subgraph pipeline [Pipeline]
        P1["Preprocess"]
        P2["EDA"]
        P3["Knowledge graph"]
        P4["Train 4 models"]
        P5["Evaluate RQ1-3"]
    end
    subgraph outputs [Outputs]
        ART["models / figures / JSON"]
        APP["RipeSense app"]
    end
    DS --> P1 --> P2 --> P3 --> P4 --> P5 --> ART
    P5 --> APP
    LIT --> P3
```

---

## Key results (held-out test set)

| Metric | Value |
|--------|-------|
| **Best model** | KG-augmented XGBoost (`kg_xgb`) |
| **Test macro-F1** | **0.9932** (99.32%) |
| **Test accuracy** | 0.9932 |
| **KG rules accepted** | 13 / 15 |
| **RQ1 — McNemar (XGB baseline vs KG)** | p = 0.522 (not significant at α = 0.05) |
| **RQ2 — SHAP rule alignment (kg_rf)** | 0.33 (humidity rules most aligned) |
| **RQ3 — Missing data (20%)** | ≥ 80% of clean macro-F1 retained |
| **RQ3 — Noise (5%)** | Falls below 80% threshold (deployment risk) |

Full metrics: `outputs/results/model_results.json`

---

## What the pipeline produces

`py -m src.run_pipeline` runs six phases end to end and overwrites everything in
`outputs/`:

| Phase | Module | Writes |
|-------|--------|--------|
| 1 | `data_loader.py` | Cleaned splits, `scaler.pkl` |
| 2 | `eda.py` | Distribution and correlation figures |
| 3 | `knowledge_graph.py`, `kg_features.py` | Validated rules, `kg_generator.json` |
| 4 | `train.py` | `baseline_rf`, `kg_rf`, `baseline_xgb`, `kg_xgb`, `best_model` |
| 5 | `evaluate.py` | `model_results.json`, SHAP importances, McNemar tests |
| 6 | `robustness.py` | `robustness.json`, degradation curves |

The trained models are committed to this repository (~53 MB), so the dashboard
runs from a fresh clone without training anything first.

---

## Research questions → code mapping

| RQ | Question | Implementation | Output files |
|----|----------|----------------|--------------|
| **RQ1** | Does KG integration improve prediction? | Four-model ablation + McNemar test | `model_results.json`, confusion matrices |
| **RQ2** | Are SHAP rankings aligned with agronomic rules? | SHAP TreeExplainer + alignment score | `rq2_alignment.json`, `shap_importance_*.json` |
| **RQ3** | Robustness under noise / missing / sensor failure? | Controlled input degradation | `robustness.json`, `robustness_curves.png` |

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the full protocol.

---

## Repository structure

```
RipeSense/
├── README.md                 # This file
├── LICENSE                   # MIT
├── requirements.txt          # Python dependencies
├── run_app.py                # ▶ RUN THIS — starts the dashboard
├── app.py                    # Streamlit RipeSense dashboard (launched by run_app.py)
├── src/                      # Pipeline source code (library modules)
│   ├── run_pipeline.py       # ▶ RUN THIS to reproduce results (Phases 1–6)
│   ├── config.py             # Paths, seeds, hyperparameters
│   ├── data_loader.py        # Load, validate, scale data
│   ├── eda.py                # Exploratory analysis + figures
│   ├── knowledge_graph.py    # NetworkX graph construction
│   ├── kg_features.py        # Rule validation + feature engineering
│   ├── train.py              # Model training (RF / XGBoost)
│   ├── evaluate.py           # Metrics, SHAP, McNemar
│   ├── robustness.py         # Noise / missing / failure tests
│   └── decision_support.py   # Inference + recommendations
├── data/
│   ├── ds_34/                # Bath dataset CSVs (train/test)
│   └── kg/                   # Knowledge graph nodes/edges
├── docs/
│   ├── ARCHITECTURE.md       # System design
│   ├── EXPERIMENTS.md        # Evaluation protocol
│   ├── DATASET.md            # Dataset documentation
│   ├── APP_GUIDE.md          # Streamlit user guide
│   └── design/               # Detailed design documents (01–05)
├── scripts/
│   └── verify_setup.py       # Automated smoke test
└── outputs/                  # Generated by the pipeline, committed for review
    ├── results/              # JSON metrics and reports
    ├── figures/              # PNG plots
    └── models/               # Trained *.pkl estimators (~53 MB)
```

---

## Dataset

University of Bath banana ripeness dataset **`ds_34`**, DOI: [10.15125/BATH-01459](https://doi.org/10.15125/BATH-01459)

- **Training:** 18,819 samples | **Test:** 8,066 samples  
- **Features:** 6 BME280 sensors (internal + ambient temperature, humidity, pressure)  
- **Labels:** Ripeness stages 1–5 (perfectly balanced, 20% each class)

Details: [docs/DATASET.md](docs/DATASET.md)

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layered system design and module responsibilities |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Six-phase pipeline and evaluation protocol |
| [docs/DATASET.md](docs/DATASET.md) | Feature definitions, ranges, exclusions |
| [docs/APP_GUIDE.md](docs/APP_GUIDE.md) | Streamlit pages and demo script |
| [docs/design/](docs/design/) | Original design documents (workflow, architecture, implementation plan) |

---

## Verification

After installation, run the automated smoke test:

```bash
py scripts/verify_setup.py
```

This checks imports, data files, pipeline outputs, and Live Prediction inference.

---

## Publishing to GitHub

From inside the `RipeSense/` folder:

```bash
git init
git add .
git commit -m "Add reproducible RipeSense pipeline, evaluation artefacts, and decision-support app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/RipeSense-banana-ripeness.git
git push -u origin main
```

Suggested repository name: **`RipeSense-banana-ripeness`** or **`knowledge-integrated-banana-ripeness`**

---

## Professor submission checklist

| Item | Location |
|------|----------|
| GitHub repository URL | Submit link to this repo |
| Reproduction commands | Start here section at the top |
| Dataset citation | [docs/DATASET.md](docs/DATASET.md) — DOI 10.15125/BATH-01459 |
| Live demo | `py run_app.py` → Live Prediction page |
| Screencast | Follow [docs/APP_GUIDE.md](docs/APP_GUIDE.md) demo script (8–12 min) |
| Written dissertation | Submitted separately via Turnitin (not in this repo) |

---

## Licence

MIT Licence — see [LICENSE](LICENSE).

## Citation

If you use this code or methodology, please cite the Bath dataset:

> Callaghan, K. & Martinez Hernandez, U. (2025). *Dataset for Low-Cost, Multi-Sensor Non-Destructive Banana Ripeness Estimation Using Machine Learning.* University of Bath Research Data Archive. https://doi.org/10.15125/BATH-01459
