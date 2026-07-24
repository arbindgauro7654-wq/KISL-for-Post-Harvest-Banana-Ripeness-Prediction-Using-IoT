# Dataset documentation — Bath `ds_34`

## Source and citation

| Field | Value |
|-------|-------|
| **Name** | University of Bath banana ripeness dataset, subset `ds_34` |
| **DOI** | [10.15125/BATH-01459](https://doi.org/10.15125/BATH-01459) |
| **Authors** | Callaghan, K. & Martinez Hernandez, U. (2025) |
| **Licence** | Open access — University of Bath Research Data Archive |
| **Local path** | `data/ds_34/` |

## Files

| File | Rows | Description |
|------|------|-------------|
| `ds_34_x_train.csv` | 18,819 | Training features |
| `ds_34_y_train.csv` | 18,819 | Training labels |
| `ds_34_x_test.csv` | 8,066 | Test features |
| `ds_34_y_test.csv` | 8,066 | Test labels |

The train/test partition is **fixed by the dataset authors** and is not modified by this project.

---

## Features (modelling set)

Six **BME280** environmental sensor channels:

| Feature | Description | Unit |
|---------|-------------|------|
| `Temp-int` | Internal temperature | °C |
| `Humid-int` | Internal relative humidity | %RH |
| `Press-int` | Internal atmospheric pressure | hPa |
| `Temp-ext` | Ambient (external) temperature | °C |
| `Humid-ext` | Ambient relative humidity | %RH |
| `Press-ext` | Ambient pressure | hPa |

### Training-set statistics (reference)

| Feature | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| Temp-int | 9.44 | 23.95 | 15.83 | 2.82 |
| Humid-int | 48.75 | 81.61 | 63.26 | 8.05 |
| Press-int | 937.15 | 1018.43 | 990.98 | 13.31 |
| Temp-ext | 8.84 | 26.64 | 15.48 | 3.05 |
| Humid-ext | 23.96 | 91.03 | 56.73 | 11.69 |
| Press-ext | 937.30 | 1018.73 | 991.38 | 13.26 |

---

## Target variable

| Field | Value |
|-------|-------|
| **Name** | Ripeness stage |
| **Type** | Integer 1–5 (ordinal) |
| **Stage 1** | Green, hard — long storage |
| **Stage 2–3** | Turning yellow-green — controlled ripening |
| **Stage 4** | Yellow — peak eating quality |
| **Stage 5** | Yellow with brown spots — over-ripe |

### Class balance

Each stage comprises **exactly 20.0%** of training and test rows. **SMOTE oversampling is not applied.**

| Stage | Train n | Test n |
|-------|---------|--------|
| 1 | 3,764 | 1,613 |
| 2 | 3,764 | 1,613 |
| 3 | 3,764 | 1,613 |
| 4 | 3,764 | 1,613 |
| 5 | 3,763 | 1,614 |

---

## Excluded modalities

The raw CSV files contain additional columns **not used** in this project:

| Category | Columns | Reason for exclusion |
|----------|---------|---------------------|
| Gas sensors | TGS20, TGS02, SGP | Focus on low-cost BME280 only |
| Spectral | SpA410–SpL940 | Higher hardware cost; SpR610 is identically zero (dead channel) |

This keeps the experiment focused on **deployable, low-cost IoT sensors** plus knowledge-graph integration.

---

## Preprocessing

1. **Range validation:** Temperature 5–35 °C, humidity 20–100 %RH, pressure 900–1050 hPa  
2. **Outlier screening:** |z| > 4 flagged (pressure only: 16 readings; retained)  
3. **Scaling:** Min–max normalisation fitted on **training set only**  
4. **Label encoding:** Stages 1–5 mapped to 0–4 internally for scikit-learn compatibility  

Scaler persisted as `outputs/models/scaler.pkl`.

---

## Dataset selection rationale

The Bath `ds_34` subset was selected because it satisfies:

1. **Topical fit** — banana ripeness as the label  
2. **Reproducibility** — fixed train/test partition  
3. **Open licence** — no primary data collection required  
4. **Sensor alignment** — includes BME280 channels used in low-cost IoT prototypes  
5. **Scope alignment** — tabular data suitable for knowledge-graph feature engineering  

Alternative sources considered (Mendeley cold-storage, image corpora, hyperspectral datasets) were excluded due to label mismatch, GPU requirements, or hardware cost. See `docs/design/05_data_story_and_dataset_rationale.md` for full comparison.

---

## Data quality checks (reference run)

- Range violations on training set: **0** for all six sensors  
- Minimum class fraction: **0.200**  
- SMOTE applied: **false**  
- Train/test leakage: **none** (scaler fit on train only)

Report: `outputs/results/data_report.json`
