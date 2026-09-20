"""Phase 1 - Data acquisition and preprocessing (docs/03).

Loads the Bath ds_34 split, restricts to the six BME280 features, performs
range validation and z-score outlier flagging, and fits a min-max scaler on the
training set only (no leakage). Class balance is reported; because the dataset
is already perfectly balanced (docs/05), SMOTE is not applied.

Viva tip: emphasise train-only scaler fit — prevents information leakage from
test set into model inputs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np       # Used for: z-score outlier counts
import pandas as pd      # Used for: loading Bath CSV splits as DataFrames
from sklearn.preprocessing import MinMaxScaler  # Used for: scale sensors to [0,1]

from . import config as C


@dataclass
class DataBundle:
    """Used for: passing raw + scaled train/test frames and scaler to all pipeline phases."""

    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train_scaled: pd.DataFrame
    X_test_scaled: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: MinMaxScaler
    report: dict = field(default_factory=dict)


def _read_split(split: str) -> tuple[pd.DataFrame, pd.Series]:
    """Used for: loading ds_34_x_{train|test}.csv and matching labels (Bath dataset)."""
    x_path = os.path.join(C.DATA_DIR, f"ds_34_x_{split}.csv")
    y_path = os.path.join(C.DATA_DIR, f"ds_34_y_{split}.csv")
    X = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)
    # The label column header is "0" but values are stages 1-5 (docs/05);
    # select by position to be safe.
    y_series = y.iloc[:, 0].astype(int)
    y_series.name = C.LABEL_NAME
    # Restrict to the six BME280 features (project scope).
    X = X[C.SENSOR_FEATURES].copy()
    return X, y_series


def _range_validation(X: pd.DataFrame) -> dict:
    """Used for: QA report — count readings outside VALID_RANGES (does not drop them)."""
    flags = {}
    for col, (lo, hi) in C.VALID_RANGES.items():
        out = ((X[col] < lo) | (X[col] > hi)).sum()
        flags[col] = int(out)
    return flags


def _zscore_outliers(X: pd.DataFrame, z: float = 4.0) -> dict:
    """Used for: EDA report — flag extreme values (|z| > 4), not used to filter rows."""
    counts = {}
    for col in X.columns:
        mu, sd = X[col].mean(), X[col].std(ddof=0)
        if sd == 0:
            counts[col] = 0
            continue
        counts[col] = int((np.abs((X[col] - mu) / sd) > z).sum())
    return counts


def load_data() -> DataBundle:
    """Used for: Phase 1 entry point — called by run_pipeline and produces scaler.pkl."""
    X_train_raw, y_train = _read_split("train")
    X_test_raw, y_test = _read_split("test")

    report: dict = {
        "n_train": int(len(X_train_raw)),
        "n_test": int(len(X_test_raw)),
        "features": list(C.SENSOR_FEATURES),
        "class_distribution_train": {
            int(k): int(v) for k, v in y_train.value_counts().sort_index().items()
        },
        "class_distribution_test": {
            int(k): int(v) for k, v in y_test.value_counts().sort_index().items()
        },
        "range_violations_train": _range_validation(X_train_raw),
        "zscore_outliers_train": _zscore_outliers(X_train_raw),
    }

    # Used for: SMOTE decision — dataset already 20% per class, so no resampling.
    frac = y_train.value_counts(normalize=True)
    report["min_class_fraction"] = float(frac.min())
    report["smote_applied"] = bool(frac.min() < 0.10)

    # Used for: MinMaxScaler — fit on TRAIN only, transform TEST with same bounds (no leakage).
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=C.SENSOR_FEATURES,
        index=X_train_raw.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=C.SENSOR_FEATURES,
        index=X_test_raw.index,
    )

    return DataBundle(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        report=report,
    )


def validate_sensor_row(values: dict) -> dict:
    """Used for: Decision Support UI — show in-range flags for manual/CSV sensor input."""
    out = {}
    for col, (lo, hi) in C.VALID_RANGES.items():
        v = float(values[col])
        out[col] = {
            "value": v,
            "in_range": lo <= v <= hi,
            "lo": lo,
            "hi": hi,
        }
    return out


def feature_summary(X: pd.DataFrame) -> pd.DataFrame:
    """Used for: slider defaults in app + feature_summary in model_results.json."""
    return pd.DataFrame(
        {
            "min": X.min(),
            "max": X.max(),
            "mean": X.mean(),
            "std": X.std(ddof=0),
        }
    ).round(3)
