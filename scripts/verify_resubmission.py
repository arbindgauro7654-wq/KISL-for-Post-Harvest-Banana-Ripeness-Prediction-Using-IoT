"""Smoke tests for resubmission artefact requirements."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib

from src import config as C
from src.decision_support import predict_compare, preprocess_preview
from src.kg_features import KGFeatureGenerator


def main() -> int:
    errors: list[str] = []
    model_dir = ROOT / "outputs" / "models"
    for name in ("scaler.pkl", "kg_generator.json", "baseline_xgb.pkl", "kg_xgb.pkl"):
        if not (model_dir / name).exists():
            errors.append(f"Missing {name}")

    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1

    scaler = joblib.load(model_dir / "scaler.pkl")
    gen = KGFeatureGenerator.load(model_dir / "kg_generator.json")
    models = {
        n: joblib.load(model_dir / f"{n}.pkl")
        for n in ("baseline_rf", "kg_rf", "baseline_xgb", "kg_xgb")
    }
    sample = {
        "Temp-int": 16.0, "Humid-int": 63.0, "Press-int": 991.0,
        "Temp-ext": 15.5, "Humid-ext": 57.0, "Press-ext": 991.0,
    }
    cmp = predict_compare(models, scaler, gen, sample, algorithm="xgb")
    for key in ("baseline", "kg", "kg_features", "preprocess", "disagree"):
        if key not in cmp:
            errors.append(f"predict_compare missing {key}")

    prev = preprocess_preview(scaler, sample)
    if not prev.get("rows"):
        errors.append("preprocess_preview empty")

    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    for needle in ("System Architecture", "Decision Support", "predict_compare"):
        if needle not in app_text:
            errors.append(f"app.py missing {needle}")

    arch = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    if "Decision Support" not in arch and "decision_support" not in arch:
        errors.append("ARCHITECTURE.md not updated")

    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1

    print("OK: resubmission artefact checks passed")
    print(json.dumps({
        "baseline_stage": cmp["baseline"]["predicted_stage"],
        "kg_stage": cmp["kg"]["predicted_stage"],
        "disagree": cmp["disagree"],
        "kg_feature_cols": len(cmp["kg_features"].columns),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
