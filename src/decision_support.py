"""Phase 6 - Decision-support logic (docs/03).

Maps a prediction plus the KG rules that fired into a plain-language storage
recommendation. Used by both the pipeline demo and the Streamlit app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .kg_features import KGFeatureGenerator
from .train import decode

# Map each rule object/concept to actionable advice.
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

STAGE_ADVICE = {
    1: "Green and starchy - suitable for long storage / shipping.",
    2: "Beginning to turn - hold or begin controlled ripening.",
    3: "Half ripe - move toward display soon.",
    4: "Ripe - sell and consume now for best quality.",
    5: "Over-ripe - use immediately (baking/processing) to avoid waste.",
}


def predict_one(model, scaler, generator: KGFeatureGenerator,
                sensor_values: dict, is_kg: bool = True) -> dict:
    """Predict ripeness for a single manual reading and explain it."""
    raw = pd.DataFrame([sensor_values])[C.SENSOR_FEATURES]
    scaled = pd.DataFrame(scaler.transform(raw), columns=C.SENSOR_FEATURES)
    if is_kg:
        kg = generator.transform(raw)
        feats = pd.concat([scaled, kg], axis=1)
    else:
        feats = scaled

    pred_enc = int(model.predict(feats)[0])
    stage = int(decode([pred_enc])[0])
    proba = model.predict_proba(feats)[0]
    confidence = float(np.max(proba))

    fired = generator.fired_rules(sensor_values) if is_kg else []
    advice = [STAGE_ADVICE[stage]]
    seen = set()
    for r in fired:
        obj = r.get("object")
        if obj in ADVICE_BY_OBJECT and obj not in seen:
            advice.append(ADVICE_BY_OBJECT[obj] + f"  (rule {r['rule_id']} {r['source']})")
            seen.add(obj)

    return {
        "predicted_stage": stage,
        "stage_label": C.STAGE_LABELS[stage],
        "confidence": round(confidence, 4),
        "class_probabilities": {int(s): round(float(p), 4)
                                 for s, p in zip(C.RIPENESS_STAGES, proba)},
        "fired_rules": fired,
        "recommendations": advice,
    }
