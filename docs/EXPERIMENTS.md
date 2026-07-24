# Experiment protocol

This document describes the six-phase evaluation pipeline implemented in `src/run_pipeline.py` and maps each phase to research questions (RQ1–RQ3).

---

## Phase overview

| Phase | Module | Purpose | Key outputs |
|-------|--------|---------|-------------|
| 1 | `data_loader.py` | Load Bath `ds_34`, validate ranges, min–max scale (train-only) | `data_report.json`, `scaler.pkl` |
| 2 | `eda.py` | Class distribution, correlation, per-stage sensor stats | `eda_summary.json`, EDA figures |
| 3 | `knowledge_graph.py`, `kg_features.py` | Build KG, validate rules, engineer features | `validated_rules.json`, `kg_generator.json`, `data/kg/*.csv` |
| 4 | `train.py` | Train 4 models with 5-fold CV (macro-F1) | `baseline_rf.pkl`, `kg_rf.pkl`, `baseline_xgb.pkl`, `kg_xgb.pkl`, `best_model.pkl` |
| 5 | `evaluate.py`, `robustness.py` | Metrics, McNemar, SHAP, degradation tests | `model_results.json`, `rq2_alignment.json`, `robustness.json` |
| 6 | `decision_support.py`, `app.py` | Inference demo + Streamlit UI | `decision_support_demo.json` |

**Run all phases:**

```bash
py -m src.run_pipeline
```

Expected runtime: **~4–5 minutes** on a standard laptop CPU (no GPU required).

---

## Models (four-model ablation)

| ID | Algorithm | Features | Purpose |
|----|-----------|----------|---------|
| A | Random Forest | 6 sensors | Baseline |
| B | Random Forest | 6 sensors + 26 KG features | KG effect within RF |
| C | XGBoost | 6 sensors | Baseline |
| D | XGBoost | 6 sensors + 26 KG features | KG effect within XGB (best overall) |

Hyperparameter tuning: `GridSearchCV`, 5 folds, scoring = **macro-F1**, `random_state = 42`.

---

## RQ1 — Integration (prediction accuracy)

**Question:** Does adding knowledge-graph features improve multi-class ripeness prediction?

**Metrics (test set):**
- Accuracy, macro-F1, weighted precision, weighted recall

**Statistical test:** McNemar test comparing baseline vs KG on identical test rows (α = 0.05).

**Expected results (reference run):**

| Model | Test macro-F1 |
|-------|---------------|
| baseline_rf | 0.9924 |
| kg_rf | 0.9906 |
| baseline_xgb | 0.9927 |
| **kg_xgb** | **0.9932** |

| Comparison | p-value | Significant? |
|------------|---------|--------------|
| RF baseline vs KG | ~0.054 | No (borderline) |
| XGB baseline vs KG | ~0.522 | No |

**Interpretation:** Sensors alone already separate ripeness stages strongly on this benchmark. KG features do not produce statistically significant accuracy gains; value is primarily in interpretability and decision support.

---

## RQ2 — Interpretability (SHAP + rule alignment)

**Question:** Are SHAP feature rankings aligned with known agronomic rules?

**Method:**
1. SHAP TreeExplainer on 500 random test samples
2. Compare directional KG rule flags against median SHAP importance
3. Compute alignment score = fraction of rules exceeding median

**Expected results:**
- Alignment score (kg_rf): **0.33**
- Humidity-related rules (R5, R6, R7) most frequently aligned
- Raw BME280 sensors dominate global SHAP rankings (expected — continuous features subsume binary flags)

Output: `outputs/results/rq2_alignment.json`, `shap_importance_*.json`, SHAP figures.

---

## RQ3 — Robustness (deployment stress tests)

**Question:** At what noise or missing-sensor rates does macro-F1 fall below 80% of clean performance?

**Degradations applied to raw test sensors (before rescaling):**

| Condition | Levels |
|-----------|--------|
| Gaussian noise | 5%, 10%, 20% of per-feature train range |
| Random missingness | 5%, 10%, 20% (median imputation) |
| Dual sensor failure | Temp-int + Temp-ext set to train median |

**Expected findings:**

| Condition | kg_rf (% of clean) | Notes |
|-----------|-------------------|-------|
| Clean | 100% | Reference |
| Noise 5% | ~68% | **Below 80% threshold** |
| Missing 20% | ~80% | Tolerant |
| Dual temp failure | ~84% | KG outperforms baseline (~73%) |

Output: `outputs/results/robustness.json`, `outputs/figures/robustness_curves.png`

---

## Knowledge-graph rule validation

Fifteen candidate rules from post-harvest literature. Each rule must pass **both**:

1. **Activation rate** ≥ 5% on training data
2. **Chi-squared** association with ripeness label, p < 0.05

**Reference outcome:** 13 accepted, 2 rejected (R13 warm+dry, R15 cool+humid — too rare in training).

Each accepted rule generates:
- Binary flag (`flag_R*`)
- Continuous risk score (`risk_R*`, capped at 5.0)
- Aggregate `kg_violation_count`

---

## Output artefact index

| File | Description |
|------|-------------|
| `outputs/results/model_results.json` | Complete results bundle |
| `outputs/results/model_results.txt` | Human-readable summary |
| `outputs/results/validated_rules.json` | Rule validation log |
| `outputs/results/eda_summary.json` | EDA statistics + quartiles |
| `outputs/results/data_report.json` | Preprocessing report |
| `outputs/results/robustness.json` | RQ3 degradation results |
| `outputs/results/rq2_alignment.json` | RQ2 alignment details |
| `outputs/results/decision_support_demo.json` | Sample predictions |
| `outputs/figures/class_distribution.png` | Class balance plot |
| `outputs/figures/correlation_heatmap.png` | Feature correlations |
| `outputs/figures/sensor_by_stage.png` | Per-stage box plots |
| `outputs/figures/model_comparison.png` | Model bar chart |
| `outputs/figures/confusion_*.png` | Confusion matrices (×4) |
| `outputs/figures/shap_importance_*.png` | SHAP plots (×2) |
| `outputs/figures/robustness_curves.png` | Robustness line charts |
| `outputs/models/*.pkl` | Trained models (regenerate locally) |
| `outputs/models/scaler.pkl` | Fitted min–max scaler |
| `outputs/models/kg_generator.json` | Serialised KG feature generator |

---

## Reproducibility controls

- Fixed random seed: **42** (`src/config.py`)
- Author-defined train/test split (unchanged from Bath dataset)
- Scaler and KG generator fitted on **training data only**
- Hyperparameters selected via cross-validation only (no test-set peeking)
- All configuration centralised in `src/config.py`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No results found` in Streamlit | Run `py -m src.run_pipeline` first |
| Missing `*.pkl` files | Pipeline not completed; re-run Phase 4–5 |
| SHAP slow or fails | Falls back to impurity importance (logged in JSON) |
| Pipeline >10 min | Normal on older CPUs; nested parallelism disabled for Windows |
