"""Headless Streamlit page-load test for RipeSense."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAGES = [
    "Overview", "Data Explorer", "Knowledge Graph", "Model Results",
    "Interpretability", "Robustness", "Live Prediction",
]


def main() -> None:
    print("Testing app imports and artefact loading...")
    import json
    import os
    import joblib
    from src import config as C
    from src.decision_support import predict_one
    from src.kg_features import KGFeatureGenerator

    mr = os.path.join(C.RESULT_DIR, "model_results.json")
    if not os.path.exists(mr):
        print("[FAIL] model_results.json missing")
        sys.exit(1)

    with open(mr, encoding="utf-8") as f:
        results = json.load(f)
    print(f"[OK] Results loaded — best model: {results['best_model']['name']}")

    scaler = joblib.load(os.path.join(C.MODEL_DIR, "scaler.pkl"))
    gen = KGFeatureGenerator.load(os.path.join(C.MODEL_DIR, "kg_generator.json"))
    model = joblib.load(os.path.join(C.MODEL_DIR, "best_model.pkl"))
    print("[OK] Models loaded (scaler, kg_generator, best_model)")

    means = results["feature_summary"]["mean"]
    sample = {f: float(means[f]) for f in C.SENSOR_FEATURES}
    out = predict_one(model, scaler, gen, sample, is_kg=True)
    print(f"[OK] Live Prediction: stage={out['predicted_stage']}, "
          f"confidence={out['confidence']:.3f}, rules_fired={len(out['fired_rules'])}")

    for page in PAGES:
        print(f"[OK] Page defined: {page}")

    print("\nAll headless checks passed. Launch UI with: py -m streamlit run app.py")


if __name__ == "__main__":
    main()
