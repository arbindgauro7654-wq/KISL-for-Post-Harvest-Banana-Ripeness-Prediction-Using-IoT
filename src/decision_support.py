"""Phase 6 - Decision-support logic (docs/03).

Maps predictions plus fired KG rules into plain-language storage advice.
Supports side-by-side baseline vs KG comparison for the Streamlit app.

Viva tip: this is offline inference replay — same path as training but on one row;
predict_compare is the key demo function for the 5-min viva walkthrough.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data_loader import validate_sensor_row
from .kg_features import KGFeatureGenerator
from .train import decode

# Used for: map fired rule objects → human-readable post-harvest storage advice
ADVICE_BY_OBJECT = {
    "ripening": "Ripening is being driven by current temperature - adjust "
                "storage temperature to control speed.",
    "moisture_loss": "Humidity is affecting moisture loss - manage ventilation "
                     "and humidity to protect fruit quality and weight.",
    "rapid_quality_loss": "Warm and dry conditions detected - act quickly: cool "
                          "the store and raise humidity to avoid rapid spoilage.",
    "fast_ripening": "Both internal and ambient temperatures are high - expect "
                     "fast ripening; prioritise dispatch or cool storage.",
    "long_shelf_life": "Cool, humid conditions favour extended shelf life - safe "
                       "to hold stock longer.",
    "atmosphere_state": "Atmospheric pressure shift noted - monitor sealed-store "
                        "conditions.",
}

# Used for: stage-specific guidance shown alongside KG rule explanations
STAGE_ADVICE = {
    1: "Green and starchy - suitable for long storage / shipping.",
    2: "Beginning to turn - hold or begin controlled ripening.",
    3: "Half ripe - move toward display soon.",
    4: "Ripe - sell and consume now for best quality.",
    5: "Over-ripe - use immediately (baking/processing) to avoid waste.",
}


def _raw_frame(sensor_values: dict) -> pd.DataFrame:
    """Used for: wrap six sensor dict values into a one-row DataFrame for sklearn."""
    return pd.DataFrame([sensor_values])[C.SENSOR_FEATURES]


def _scale(scaler, raw: pd.DataFrame) -> pd.DataFrame:
    """Used for: apply saved MinMaxScaler (same as training — no refitting)."""
    return pd.DataFrame(scaler.transform(raw), columns=C.SENSOR_FEATURES)


def _build_feature_matrix(scaler, generator: KGFeatureGenerator,
                          sensor_values: dict, is_kg: bool) -> pd.DataFrame:
    """Used for: 6 scaled columns (baseline) or 6 + KG columns (kg model input)."""
    raw = _raw_frame(sensor_values)
    scaled = _scale(scaler, raw)
    if is_kg:
        kg = generator.transform(raw)
        return pd.concat([scaled, kg], axis=1)
    return scaled


def _predict_from_matrix(model, feats: pd.DataFrame) -> dict:
    """Used for: sklearn predict + predict_proba → stage label and confidence."""
    pred_enc = int(model.predict(feats)[0])
    stage = int(decode([pred_enc])[0])
    proba = model.predict_proba(feats)[0]
    return {
        "predicted_stage": stage,
        "stage_label": C.STAGE_LABELS[stage],
        "confidence": round(float(np.max(proba)), 4),
        "class_probabilities": {
            int(s): round(float(p), 4) for s, p in zip(C.RIPENESS_STAGES, proba)
        },
    }


def _recommendations(stage: int, fired: list[dict]) -> list[str]:
    """Used for: combine stage advice + one line per unique fired rule object."""
    advice = [STAGE_ADVICE[stage]]
    seen = set()
    for r in fired:
        obj = r.get("object")
        if obj in ADVICE_BY_OBJECT and obj not in seen:
            advice.append(
                ADVICE_BY_OBJECT[obj] + f"  (rule {r['rule_id']} {r['source']})"
            )
            seen.add(obj)
    return advice


def preprocess_preview(scaler, sensor_values: dict) -> dict:
    """Used for: Data Explorer + Decision Support — show raw vs scaled + in-range flags."""
    raw = _raw_frame(sensor_values)
    scaled = _scale(scaler, raw)
    rows = []
    for feat in C.SENSOR_FEATURES:
        rows.append({
            "feature": feat,
            "raw": round(float(raw[feat].iloc[0]), 4),
            "scaled": round(float(scaled[feat].iloc[0]), 4),
            **validate_sensor_row(sensor_values)[feat],
        })
    return {"rows": rows, "all_in_range": all(r["in_range"] for r in rows)}


def kg_feature_table(generator: KGFeatureGenerator, sensor_values: dict) -> pd.DataFrame:
    """Used for: Decision Support table — flag_*, risk_*, kg_violation_count for this row."""
    raw = _raw_frame(sensor_values)
    kg = generator.transform(raw)
    return kg.round(4)


def local_top_features(model, feats: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Used for: quick local explanation — top tree impurity features (not full SHAP)."""
    names = list(feats.columns)
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(names):
        return []
    pairs = sorted(zip(names, importances), key=lambda x: -x[1])[:top_n]
    return [{"feature": n, "importance": round(float(v), 4)} for n, v in pairs]


def predict_one(model, scaler, generator: KGFeatureGenerator,
                sensor_values: dict, is_kg: bool = True) -> dict:
    """Used for: single-model inference path (pipeline demo JSON + legacy UI)."""
    feats = _build_feature_matrix(scaler, generator, sensor_values, is_kg)
    out = _predict_from_matrix(model, feats)
    fired = generator.fired_rules(sensor_values) if is_kg else []
    out["fired_rules"] = fired
    out["recommendations"] = _recommendations(out["predicted_stage"], fired)
    return out


def predict_compare(models: dict, scaler, generator: KGFeatureGenerator,
                    sensor_values: dict, algorithm: str = "xgb") -> dict:
    """Used for: viva demo — side-by-side baseline vs KG prediction + explainability."""
    algo = algorithm.lower()
    base_key = f"baseline_{algo}"
    kg_key = f"kg_{algo}"
    if base_key not in models or kg_key not in models:
        raise KeyError(f"Models {base_key} and {kg_key} must be loaded")

    base_feats = _build_feature_matrix(scaler, generator, sensor_values, False)
    kg_feats = _build_feature_matrix(scaler, generator, sensor_values, True)

    baseline = _predict_from_matrix(models[base_key], base_feats)
    kg = _predict_from_matrix(models[kg_key], kg_feats)
    fired = generator.fired_rules(sensor_values)
    kg["fired_rules"] = fired
    kg["recommendations"] = _recommendations(kg["predicted_stage"], fired)

    disagree = baseline["predicted_stage"] != kg["predicted_stage"]
    return {
        "algorithm": algo.upper(),
        "baseline": baseline,
        "kg": kg,
        "disagree": disagree,
        "kg_features": kg_feature_table(generator, sensor_values),
        "preprocess": preprocess_preview(scaler, sensor_values),
        "baseline_local_features": local_top_features(
            models[base_key], base_feats),
        "kg_local_features": local_top_features(models[kg_key], kg_feats),
    }
