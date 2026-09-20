"""Phase 4 - Model training and the four-model ablation (docs/03).

Trains:
  A = Random Forest baseline (6 sensors)
  B = Random Forest + KG features
  C = XGBoost baseline (6 sensors)
  D = XGBoost + KG features

Hyper-parameters are tuned with 5-fold cross-validation scored on macro-F1.
Labels (stages 1-5) are encoded to 0-4 for compatibility with both estimators.

Viva tip: four-model ablation isolates KG effect within each algorithm family (RQ1).
"""
from __future__ import annotations

import os

import joblib       # Used for: save/load .pkl model artefacts for the app
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Used for: bagging baseline + SHAP (kg_rf)
from sklearn.model_selection import GridSearchCV     # Used for: 5-fold CV hyperparameter search
from xgboost import XGBClassifier                    # Used for: best accuracy (gradient boosted trees)

from . import config as C


def _encode(y: pd.Series) -> np.ndarray:
    """Used for: map ripeness stages 1–5 → 0–4 (sklearn/XGBoost class indices)."""
    return (y.values - 1).astype(int)


def decode(pred_enc: np.ndarray) -> np.ndarray:
    """Used for: map model output 0–4 → ripeness stages 1–5 for display/advice."""
    return (np.asarray(pred_enc) + 1).astype(int)


def _make_rf() -> RandomForestClassifier:
    """Used for: ensemble of decision trees — stable baseline, good for SHAP (RQ2)."""
    # n_jobs=1 here — GridSearchCV parallelises folds; avoids Windows oversubscription.
    return RandomForestClassifier(random_state=C.RANDOM_SEED, n_jobs=1)


def _make_xgb() -> XGBClassifier:
    """Used for: gradient-boosted trees — highest test macro-F1 in this project."""
    return XGBClassifier(
        random_state=C.RANDOM_SEED,
        objective="multi:softprob",
        num_class=len(C.RIPENESS_STAGES),
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=1,
    )


def _tune(estimator, grid, X, y_enc):
    """Used for: GridSearchCV with macro-F1 — picks best hyperparameters per model."""
    gs = GridSearchCV(
        estimator, grid, scoring=C.SCORING, cv=C.CV_FOLDS, n_jobs=-1, refit=True
    )
    gs.fit(X, y_enc)
    # Refit the best estimator with all cores for fast downstream inference.
    best = gs.best_estimator_
    if hasattr(best, "n_jobs"):
        best.set_params(n_jobs=-1)
    return gs


def train_all(X_train_base, X_train_aug, y_train) -> dict:
    """Used for: train all 4 models, save .pkl files, return CV metadata for report."""
    y_enc = _encode(y_train)
    models = {}
    cv_meta = {}

    # Used for: four-model ablation — baseline (6 feat) vs KG-augmented (32 feat) × RF/XGB
    configs = [
        ("baseline_rf", _make_rf(), C.RF_GRID, X_train_base),
        ("kg_rf", _make_rf(), C.RF_GRID, X_train_aug),
        ("baseline_xgb", _make_xgb(), C.XGB_GRID, X_train_base),
        ("kg_xgb", _make_xgb(), C.XGB_GRID, X_train_aug),
    ]

    for name, est, grid, X in configs:
        gs = _tune(est, grid, X, y_enc)
        models[name] = gs.best_estimator_
        cv_meta[name] = {
            "best_params": gs.best_params_,
            "cv_best_macro_f1": round(float(gs.best_score_), 4),
            "n_features": int(X.shape[1]),
            "feature_names": list(X.columns),
        }
        joblib.dump(gs.best_estimator_, os.path.join(C.MODEL_DIR, f"{name}.pkl"))

    return {"models": models, "cv_meta": cv_meta}
