"""Central configuration: paths, seeds, feature lists, model grids and the
literature-based knowledge-graph rule specifications.

All design decisions here trace back to the documents in ``docs/``:
- six BME280 features only (docs/05)
- labels are integer ripeness stages 1-5, already perfectly balanced (docs/05)
- KG rules derived from post-harvest literature (docs/03, docs/04)
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "ds_34")
KG_DIR = os.path.join(ROOT, "data", "kg")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

for _d in (KG_DIR, OUTPUT_DIR, FIG_DIR, MODEL_DIR, RESULT_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Data schema
# --------------------------------------------------------------------------- #
# The six low-cost BME280 environmental features that define the project scope.
SENSOR_FEATURES = [
    "Temp-int",
    "Humid-int",
    "Press-int",
    "Temp-ext",
    "Humid-ext",
    "Press-ext",
]

LABEL_NAME = "ripeness_stage"  # canonical name we assign after loading
RIPENESS_STAGES = [1, 2, 3, 4, 5]

# Human-readable description of each ripeness stage (banana colour scale, docs/04).
STAGE_LABELS = {
    1: "Stage 1 - Green (hard, starchy)",
    2: "Stage 2 - Turning (more green than yellow)",
    3: "Stage 3 - Half ripe (yellow-green)",
    4: "Stage 4 - Ripe (yellow)",
    5: "Stage 5 - Over-ripe (yellow with brown spots)",
}

# Physically plausible sensor ranges for range-validation (docs/03, Phase 1).
VALID_RANGES = {
    "Temp-int": (5.0, 35.0),
    "Temp-ext": (5.0, 35.0),
    "Humid-int": (20.0, 100.0),
    "Humid-ext": (20.0, 100.0),
    "Press-int": (900.0, 1050.0),
    "Press-ext": (900.0, 1050.0),
}

# Banana chilling-injury threshold (deg C) - used as a literature-grounded
# knowledge-graph boundary (docs/04, Siddiqui et al. [1]).
CHILLING_THRESHOLD_C = 13.0
# Accelerated-ripening temperature (deg C) - Golding et al. [6].
ACCEL_RIPENING_C = 20.0

# --------------------------------------------------------------------------- #
# Knowledge-graph rule specifications (docs/03 Phase 3, docs/04)
# --------------------------------------------------------------------------- #
# Each rule becomes (after validation) up to three model features:
#   1) binary activation flag
#   2) continuous risk score (how far past the threshold)
#   3) contributes to an aggregate violation count
#
# ``q`` is the training-set quantile used to set a data-driven threshold.
# ``fixed`` (optional) is a published literature threshold; it is preferred when
# it lies within the observed feature range, otherwise the quantile is used
# (docs/03, "threshold selection").
#
# expected_dir: expected SHAP direction w.r.t. higher ripeness stage, used for
# the RQ2 interpretability alignment score.
KG_RULES = [
    # --- Temperature: the master driver of ripening rate [6] ---------------- #
    dict(id="R1", subject="Temp-int", op=">", q=0.75, fixed=ACCEL_RIPENING_C,
         predicate="accelerates", object="ripening", expected_dir="positive",
         source="[6]"),
    dict(id="R2", subject="Temp-int", op="<", q=0.25, fixed=CHILLING_THRESHOLD_C,
         predicate="slows", object="ripening", expected_dir="negative",
         source="[1]"),
    dict(id="R3", subject="Temp-ext", op=">", q=0.75, fixed=ACCEL_RIPENING_C,
         predicate="accelerates", object="ripening", expected_dir="positive",
         source="[6]"),
    dict(id="R4", subject="Temp-ext", op="<", q=0.25, fixed=None,
         predicate="slows", object="ripening", expected_dir="negative",
         source="[6]"),
    # --- Humidity: governs moisture loss / shelf life [1] ------------------- #
    dict(id="R5", subject="Humid-int", op=">", q=0.75, fixed=None,
         predicate="slows", object="moisture_loss", expected_dir="positive",
         source="[1]"),
    dict(id="R6", subject="Humid-int", op="<", q=0.25, fixed=None,
         predicate="increases", object="moisture_loss", expected_dir="negative",
         source="[1]"),
    dict(id="R7", subject="Humid-ext", op=">", q=0.75, fixed=None,
         predicate="slows", object="moisture_loss", expected_dir="positive",
         source="[1]"),
    dict(id="R8", subject="Humid-ext", op="<", q=0.25, fixed=None,
         predicate="increases", object="moisture_loss", expected_dir="negative",
         source="[1]"),
    # --- Pressure: proxy for sealed atmosphere / respiration [1] ------------ #
    dict(id="R9", subject="Press-int", op=">", q=0.75, fixed=None,
         predicate="associated_with", object="atmosphere_state",
         expected_dir="neutral", source="[1]"),
    dict(id="R10", subject="Press-int", op="<", q=0.25, fixed=None,
         predicate="associated_with", object="atmosphere_state",
         expected_dir="neutral", source="[1]"),
    dict(id="R11", subject="Press-ext", op=">", q=0.75, fixed=None,
         predicate="associated_with", object="atmosphere_state",
         expected_dir="neutral", source="[1]"),
    dict(id="R12", subject="Press-ext", op="<", q=0.25, fixed=None,
         predicate="associated_with", object="atmosphere_state",
         expected_dir="neutral", source="[1]"),
]

# Interaction rules (combine two sensors) - encode agronomic compound effects.
KG_INTERACTION_RULES = [
    dict(id="R13", left="Temp-int", left_op=">", left_q=0.75,
         right="Humid-int", right_op="<", right_q=0.25,
         predicate="causes", object="rapid_quality_loss",
         expected_dir="positive", source="[1],[6]",
         label="Warm & dry: rapid quality loss"),
    dict(id="R14", left="Temp-int", left_op=">", left_q=0.75,
         right="Temp-ext", right_op=">", right_q=0.75,
         predicate="causes", object="fast_ripening",
         expected_dir="positive", source="[6]",
         label="Both temperatures high: fast ripening"),
    dict(id="R15", left="Temp-int", left_op="<", left_q=0.25,
         right="Humid-int", right_op=">", right_q=0.75,
         predicate="supports", object="long_shelf_life",
         expected_dir="negative", source="[1]",
         label="Cool & humid: extended shelf life"),
]

# --------------------------------------------------------------------------- #
# Rule validation thresholds (docs/03 Phase 3)
# --------------------------------------------------------------------------- #
MIN_ACTIVATION_RATE = 0.05   # rule must fire on >= 5% of training rows
CHI2_ALPHA = 0.05            # chi-squared association significance level

# --------------------------------------------------------------------------- #
# Model hyper-parameter grids (kept compact for laptop-CPU runtime, docs/02)
# --------------------------------------------------------------------------- #
CV_FOLDS = 5
SCORING = "f1_macro"

# Grids are kept deliberately compact (bounded tree depth) so the full 5-fold
# CV completes in a couple of minutes on a laptop CPU (docs/02 non-functional
# requirement: "runs in minutes").
RF_GRID = {
    "n_estimators": [200],
    "max_depth": [12, 20],
    "min_samples_leaf": [1, 2],
}

XGB_GRID = {
    "n_estimators": [300],
    "max_depth": [4, 6],
    "learning_rate": [0.3],
}

# Number of test rows sampled for SHAP (speed vs. fidelity trade-off, docs/03).
SHAP_SAMPLE_SIZE = 500

# Robustness sweep settings (docs/03 Phase 5).
NOISE_LEVELS = [0.05, 0.10, 0.20]
MISSING_LEVELS = [0.05, 0.10, 0.20]
ROBUSTNESS_THRESHOLD = 0.80  # macro-F1 must stay >= 80% of clean score
