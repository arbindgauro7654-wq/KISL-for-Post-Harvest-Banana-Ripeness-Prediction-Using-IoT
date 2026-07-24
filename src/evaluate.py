"""Phase 5 - Evaluation (docs/03).

Computes RQ1 metrics (+ McNemar significance), RQ2 SHAP-based interpretability
alignment, and renders figures (confusion matrices, model comparison, SHAP
importance). Falls back to impurity-based importance if SHAP fails.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from . import config as C
from .train import _encode, decode

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------- #
# RQ1 - classification metrics
# --------------------------------------------------------------------------- #
def classification_metrics(model, X, y_true) -> dict:
    y_enc = _encode(y_true)
    pred = model.predict(X)
    return {
        "accuracy": round(float(accuracy_score(y_enc, pred)), 4),
        "macro_f1": round(float(f1_score(y_enc, pred, average="macro")), 4),
        "weighted_precision": round(float(precision_score(y_enc, pred, average="weighted", zero_division=0)), 4),
        "weighted_recall": round(float(recall_score(y_enc, pred, average="weighted", zero_division=0)), 4),
    }


def mcnemar_test(model_a, X_a, model_b, X_b, y_true) -> dict:
    """McNemar test comparing two classifiers' correctness on the same rows."""
    y_enc = _encode(y_true)
    pa = model_a.predict(X_a) == y_enc
    pb = model_b.predict(X_b) == y_enc
    b = int(np.sum(pa & ~pb))   # A right, B wrong
    c = int(np.sum(~pa & pb))   # A wrong, B right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "significant": False}
    # continuity-corrected chi-squared with 1 dof
    stat = (abs(b - c) - 1) ** 2 / n
    from scipy.stats import chi2
    p = float(chi2.sf(stat, df=1))
    return {"b": b, "c": c, "statistic": round(float(stat), 4),
            "p_value": round(p, 6), "significant": bool(p < 0.05)}


def confusion_fig(model, X, y_true, title, fname):
    y_enc = _encode(y_true)
    cm = confusion_matrix(y_enc, model.predict(X))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn",
                xticklabels=C.RIPENESS_STAGES, yticklabels=C.RIPENESS_STAGES, ax=ax)
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("True stage")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, fname), dpi=130)
    plt.close(fig)


def comparison_fig(results: dict):
    rows = []
    for name, m in results.items():
        rows.append({"model": name, **{k: m["test"][k] for k in
                    ("accuracy", "macro_f1", "weighted_precision", "weighted_recall")}})
    df = pd.DataFrame(rows).set_index("model")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    df.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on test set (baseline vs KG-augmented)")
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "model_comparison.png"), dpi=130)
    plt.close(fig)
    return df


# --------------------------------------------------------------------------- #
# RQ2 - SHAP interpretability + alignment with agronomic rules
# --------------------------------------------------------------------------- #
def shap_importance(model, X_sample: pd.DataFrame, tag: str) -> dict:
    """Mean |SHAP| per feature; falls back to impurity importance on failure."""
    importances = None
    method = "shap"
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        arr = np.array(sv)
        # multiclass shapes: (classes, n, feats) or (n, feats, classes)
        if arr.ndim == 3:
            axis_feats = 2 if arr.shape[2] == X_sample.shape[1] else 1
            mean_abs = np.abs(arr).mean(axis=tuple(i for i in range(arr.ndim) if i != axis_feats))
        else:
            mean_abs = np.abs(arr).mean(axis=0)
        importances = np.asarray(mean_abs).ravel()[: X_sample.shape[1]]
    except Exception as exc:  # pragma: no cover - environment dependent
        method = f"impurity_fallback ({type(exc).__name__})"
        importances = np.asarray(getattr(model, "feature_importances_"))

    imp = pd.Series(importances, index=X_sample.columns).sort_values(ascending=False)
    imp_dict = {k: round(float(v), 6) for k, v in imp.items()}

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(imp))))
    sns.barplot(x=imp.values, y=imp.index, hue=imp.index, palette="mako",
                legend=False, ax=ax)
    ax.set_title(f"Feature importance ({method}) - {tag}")
    ax.set_xlabel("Mean |SHAP value|" if method == "shap" else "Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, f"shap_importance_{tag}.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(C.RESULT_DIR, f"shap_importance_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"method": method, "importance": imp_dict}, f, indent=2)

    return {"method": method, "importance": imp_dict}


def rule_alignment_score(generator, importance: dict) -> dict:
    """RQ2: fraction of KG rules whose flag feature is an influential predictor.

    A rule is 'aligned' if its activation flag has non-trivial importance and the
    rule carries a directional (non-neutral) agronomic expectation.
    """
    if not importance:
        return {"alignment_score": 0.0, "checked": 0, "aligned": 0, "details": []}
    vals = np.array(list(importance.values()))
    threshold = float(np.median(vals[vals > 0])) if np.any(vals > 0) else 0.0

    details, aligned, checked = [], 0, 0
    directional = [r for r in (generator.rules + generator.interactions)
                   if r.get("expected_dir") in ("positive", "negative")]
    for r in directional:
        checked += 1
        flag = f"flag_{r['id']}"
        imp = importance.get(flag, 0.0)
        is_aligned = imp >= threshold and imp > 0
        aligned += int(is_aligned)
        details.append({
            "rule_id": r["id"],
            "expected_dir": r["expected_dir"],
            "flag_importance": round(float(imp), 6),
            "aligned": bool(is_aligned),
            "source": r.get("source"),
        })
    score = round(aligned / checked, 4) if checked else 0.0
    return {"alignment_score": score, "checked": checked, "aligned": aligned,
            "median_importance_threshold": round(threshold, 6), "details": details}
