# RipeSense application guide

## Launch

```bash
py -m streamlit run app.py
```

Open **http://localhost:8501**. Requires pipeline outputs (`py -m src.run_pipeline`).

---

## Pages

| Page | Purpose |
|------|---------|
| **Overview** | Project summary, key metrics, architecture flowchart, RQ1 findings |
| **Data Explorer** | Feature ranges, class distribution, correlation heatmap, per-stage means |
| **Knowledge Graph** | Validated/rejected rules, activation rates, chi-squared results |
| **Model Results** | Four-model comparison, McNemar tests, confusion matrix selector |
| **Interpretability** | SHAP importance plots, RQ2 rule-alignment score and details |
| **Robustness** | Noise/missing/failure curves vs 80% threshold (RQ3) |
| **Live Prediction** | Enter sensor values → ripeness stage + fired rules + storage advice |

---

## Live Prediction demo script (screencast)

Use this sequence for an **8–12 minute** demonstration:

### 1. Overview (1 min)
- Show best macro-F1 (**0.993**), KG rules (**13/15**), test samples (**8,066**)
- Briefly explain the data → KG → model → advice flow

### 2. Knowledge Graph (1 min)
- Scroll validated rules table
- Point out rejected rules R13/R15 (low activation)

### 3. Model Results (2 min)
- Show four-model comparison table
- Highlight **kg_xgb** as best model
- Show McNemar p-values (not significant — honest null result)
- Display confusion matrix for `kg_xgb`

### 4. Interpretability (1 min)
- Show SHAP plots (baseline vs KG-augmented)
- Mention alignment score **0.33** — humidity rules most aligned

### 5. Robustness (1 min)
- Show noise sensitivity (drops below 80% at 5%)
- Contrast with missing-data tolerance

### 6. Live Prediction (3–4 min)

**Demo A — Typical readings (mid-range):**
| Sensor | Value |
|--------|-------|
| Temp-int | 16.0 °C |
| Humid-int | 63.0 %RH |
| Press-int | 991.0 hPa |
| Temp-ext | 15.5 °C |
| Humid-ext | 57.0 %RH |
| Press-ext | 991.0 hPa |

Select **KG-augmented (recommended)** → Predict. Show stage, confidence, probability bars, recommendations.

**Demo B — High temperature (rule firing):**
| Sensor | Value |
|--------|-------|
| Temp-int | **22.0** °C |
| Humid-int | 55.0 %RH |
| Press-int | 991.0 hPa |
| Temp-ext | **24.0** °C |
| Humid-ext | 45.0 %RH |
| Press-ext | 991.0 hPa |

Show **fired rules** (e.g. R1/R3 temperature rules) and storage advice mentioning temperature reduction.

**Demo C — Sensor-only baseline:**
Repeat Demo B with **Sensor-only baseline** model to compare rule visibility (KG model shows fired rules; baseline does not).

### 7. Closing (30 sec)
- Reiterate: reproducible pipeline, open dataset, decision-support (not autonomous control)

---

## Screencast checklist

- [ ] Pipeline has been run (`outputs/results/model_results.json` exists)
- [ ] Browser at 1080p, Streamlit at full width
- [ ] Microphone clear; narrate each page purpose
- [ ] Live Prediction demos A and B completed
- [ ] Save recording as `[StudentID]_RipeSense_demo.mp4`
- [ ] Upload per Moodle instructions

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Red sidebar: "Run pipeline first" | Execute `py -m src.run_pipeline` |
| Model load error | Check `outputs/models/scaler.pkl` and `best_model.pkl` exist |
| Blank SHAP page | Re-run pipeline; check `shap_importance_*.json` in results |
| Slider out of range | Values constrained to training min/max automatically |

---

## Design notes

- **Human-in-the-loop:** All recommendations are advisory; operators retain control of ripening schedules
- **No cloud dependency:** Runs entirely on localhost
- **Train/serve consistency:** App loads the same `scaler.pkl` and `kg_generator.json` produced by the pipeline
