"""Phase 2 - Exploratory data analysis (docs/03).

Produces per-stage box plots, a correlation heatmap, a class-distribution chart,
per-stage statistics and quartile boundaries that inform KG thresholds.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config as C

sns.set_theme(style="whitegrid")


def run_eda(X_raw: pd.DataFrame, y: pd.Series) -> dict:
    df = X_raw.copy()
    df[C.LABEL_NAME] = y.values

    # ---- class distribution ---- #
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = y.value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                palette="YlGn", legend=False, ax=ax)
    ax.set_xlabel("Ripeness stage")
    ax.set_ylabel("Count")
    ax.set_title("Class distribution (ds_34 training set)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "class_distribution.png"), dpi=130)
    plt.close(fig)

    # ---- correlation heatmap ---- #
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = X_raw.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax)
    ax.set_title("Sensor correlation heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "correlation_heatmap.png"), dpi=130)
    plt.close(fig)

    # ---- per-stage box plots ---- #
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, C.SENSOR_FEATURES):
        sns.boxplot(data=df, x=C.LABEL_NAME, y=col, hue=C.LABEL_NAME,
                    palette="YlGn", legend=False, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("Stage")
    fig.suptitle("Sensor distributions by ripeness stage", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "sensor_by_stage.png"), dpi=130)
    plt.close(fig)

    # ---- numeric summaries ---- #
    per_stage_mean = df.groupby(C.LABEL_NAME)[C.SENSOR_FEATURES].mean().round(3)
    per_stage_std = df.groupby(C.LABEL_NAME)[C.SENSOR_FEATURES].std().round(3)
    quartiles = {
        col: {
            "q25": round(float(X_raw[col].quantile(0.25)), 3),
            "q50": round(float(X_raw[col].quantile(0.50)), 3),
            "q75": round(float(X_raw[col].quantile(0.75)), 3),
        }
        for col in C.SENSOR_FEATURES
    }

    summary = {
        "feature_ranges": {
            col: {
                "min": round(float(X_raw[col].min()), 3),
                "max": round(float(X_raw[col].max()), 3),
                "mean": round(float(X_raw[col].mean()), 3),
                "std": round(float(X_raw[col].std(ddof=0)), 3),
            }
            for col in C.SENSOR_FEATURES
        },
        "per_stage_mean": per_stage_mean.to_dict(),
        "per_stage_std": per_stage_std.to_dict(),
        "quartiles": quartiles,
        "correlation": corr.round(3).to_dict(),
    }

    with open(os.path.join(C.RESULT_DIR, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
