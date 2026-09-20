"""Phase 5 (part) - Robustness stress-testing (docs/03, RQ3).

Degrades the raw test sensors with (a) Gaussian noise, (b) random missing
values (median-imputed), and (c) a simulated sensor-pair failure, then
re-derives features (recomputing KG features from the degraded sensors) and
reports macro-F1 as a percentage of clean-data performance.

Viva tip: RQ3 asks whether models stay usable under noisy/missing IoT data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score  # Used for: macro-F1 under each degradation scenario

from . import config as C
from .kg_features import KGFeatureGenerator
from .train import _encode, decode


def _features(X_raw, scaler, generator, is_kg):
    """Used for: rebuild feature matrix after degrading raw sensors (re-scale + re-KG)."""
    scaled = pd.DataFrame(
        scaler.transform(X_raw[C.SENSOR_FEATURES]),
        columns=C.SENSOR_FEATURES, index=X_raw.index,
    )
    if not is_kg:
        return scaled
    kg = generator.transform(X_raw)
    return pd.concat([scaled, kg], axis=1)


def _macro_f1(model, X_feat, y_true_enc):
    pred = model.predict(X_feat)
    return float(f1_score(y_true_enc, pred, average="macro"))


def _add_noise(X_raw, level, ranges, rng):
    """Used for: inject Gaussian noise at 5/10/20% of each sensor's training range."""
    X = X_raw.copy()
    for col in C.SENSOR_FEATURES:
        sigma = level * ranges[col]
        X[col] = X[col] + rng.normal(0.0, sigma, size=len(X))
    return X


def _add_missing(X_raw, level, medians, rng):
    """Used for: randomly blank values then impute with training median (simple baseline)."""
    X = X_raw.copy()
    for col in C.SENSOR_FEATURES:
        mask = rng.random(len(X)) < level
        X.loc[mask, col] = medians[col]  # impute with train median
    return X


def _sensor_failure(X_raw, medians):
    """Used for: simulate both temperature sensors failing → set to train median."""
    X = X_raw.copy()
    for col in ("Temp-int", "Temp-ext"):
        X[col] = medians[col]
    return X


def run_robustness(model, scaler, generator: KGFeatureGenerator,
                   X_test_raw, y_test, is_kg, train_ranges, train_medians):
    """Used for: RQ3 — return clean F1 + degraded F1 as % of clean for Robustness page."""
    rng = np.random.default_rng(C.RANDOM_SEED)
    y_enc = _encode(y_test)

    clean = _macro_f1(model, _features(X_test_raw, scaler, generator, is_kg), y_enc)
    results = {"clean_macro_f1": round(clean, 4), "noise": {}, "missing": {}, "sensor_failure": {}}

    for lvl in C.NOISE_LEVELS:
        Xd = _add_noise(X_test_raw, lvl, train_ranges, rng)
        score = _macro_f1(model, _features(Xd, scaler, generator, is_kg), y_enc)
        results["noise"][f"{int(lvl*100)}%"] = {
            "macro_f1": round(score, 4),
            "pct_of_clean": round(100 * score / clean, 1) if clean else 0.0,
        }

    for lvl in C.MISSING_LEVELS:
        Xd = _add_missing(X_test_raw, lvl, train_medians, rng)
        score = _macro_f1(model, _features(Xd, scaler, generator, is_kg), y_enc)
        results["missing"][f"{int(lvl*100)}%"] = {
            "macro_f1": round(score, 4),
            "pct_of_clean": round(100 * score / clean, 1) if clean else 0.0,
        }

    Xd = _sensor_failure(X_test_raw, train_medians)
    score = _macro_f1(model, _features(Xd, scaler, generator, is_kg), y_enc)
    results["sensor_failure"] = {
        "macro_f1": round(score, 4),
        "pct_of_clean": round(100 * score / clean, 1) if clean else 0.0,
        "description": "Temp-int and Temp-ext set to train median (dual failure)",
    }

    return results
