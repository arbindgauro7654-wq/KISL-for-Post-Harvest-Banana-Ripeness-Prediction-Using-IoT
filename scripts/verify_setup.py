#!/usr/bin/env python3
"""Smoke test for RipeSense artefact — run from repository root.

Usage:
    py scripts/verify_setup.py
    py scripts/verify_setup.py --full   # also run Live Prediction inference
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def ok(msg: str) -> None:
    global PASS
    PASS += 1
    print(f"  [OK] {msg}")


def bad(msg: str) -> None:
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {msg}")


def check_imports() -> None:
    print("\n1. Python imports")
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
        import xgboost  # noqa: F401
        import networkx  # noqa: F401
        import shap  # noqa: F401
        import streamlit  # noqa: F401
        ok("Core dependencies importable")
    except ImportError as e:
        bad(f"Missing dependency: {e}")

    try:
        from src import config, data_loader, run_pipeline  # noqa: F401
        ok("Project modules importable")
    except ImportError as e:
        bad(f"Project import failed: {e}")


def check_data() -> None:
    print("\n2. Dataset files")
    ds = ROOT / "data" / "ds_34"
    for name in ("ds_34_x_train.csv", "ds_34_y_train.csv",
                 "ds_34_x_test.csv", "ds_34_y_test.csv"):
        p = ds / name
        if p.exists() and p.stat().st_size > 1000:
            ok(name)
        else:
            bad(f"Missing or empty: {name}")


def check_results() -> None:
    print("\n3. Pipeline results")
    results = ROOT / "outputs" / "results" / "model_results.json"
    if not results.exists():
        bad("model_results.json missing — run: py -m src.run_pipeline")
        return
    ok("model_results.json present")
    with open(results, encoding="utf-8") as f:
        data = json.load(f)
    best = data.get("best_model", {}).get("name", "")
    if best == "kg_xgb":
        ok(f"Best model: {best}")
    else:
        bad(f"Unexpected best model: {best!r} (expected kg_xgb)")
    f1 = data["models"]["kg_xgb"]["test"]["macro_f1"]
    if f1 >= 0.99:
        ok(f"kg_xgb test macro-F1 = {f1:.4f}")
    else:
        bad(f"kg_xgb macro-F1 too low: {f1:.4f}")


def check_models() -> None:
    print("\n4. Model artefacts")
    models = ROOT / "outputs" / "models"
    required = ("scaler.pkl", "kg_generator.json", "best_model.pkl",
                "baseline_rf.pkl", "kg_rf.pkl", "baseline_xgb.pkl", "kg_xgb.pkl")
    for name in required:
        p = models / name
        if p.exists():
            ok(name)
        else:
            bad(f"Missing: outputs/models/{name} — run pipeline")


def check_figures() -> None:
    print("\n5. Figures")
    figs = ROOT / "outputs" / "figures"
    expected = (
        "class_distribution.png", "correlation_heatmap.png", "sensor_by_stage.png",
        "model_comparison.png", "confusion_kg_xgb.png", "robustness_curves.png",
    )
    for name in expected:
        if (figs / name).exists():
            ok(name)
        else:
            bad(f"Missing figure: {name}")


def check_inference() -> None:
    print("\n6. Live Prediction inference")
    try:
        import joblib
        from src import config as C
        from src.decision_support import predict_one
        from src.kg_features import KGFeatureGenerator

        scaler = joblib.load(ROOT / "outputs" / "models" / "scaler.pkl")
        gen = KGFeatureGenerator.load(ROOT / "outputs" / "models" / "kg_generator.json")
        model = joblib.load(ROOT / "outputs" / "models" / "kg_rf.pkl")

        sample = {}
        mr = ROOT / "outputs" / "results" / "model_results.json"
        if mr.exists():
            with open(mr, encoding="utf-8") as f:
                fs = json.load(f).get("feature_summary", {}).get("mean", {})
            for feat in C.SENSOR_FEATURES:
                sample[feat] = float(fs.get(feat, 15.0))
        else:
            sample = {f: 15.0 for f in C.SENSOR_FEATURES}

        out = predict_one(model, scaler, gen, sample, is_kg=True)
        stage = out.get("predicted_stage")
        if 1 <= stage <= 5:
            ok(f"predict_one returned stage {stage}, confidence {out['confidence']:.3f}")
        else:
            bad(f"Invalid predicted stage: {stage}")
    except Exception as e:
        bad(f"Inference failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RipeSense smoke test")
    parser.add_argument("--full", action="store_true",
                        help="Include Live Prediction inference test")
    args = parser.parse_args()

    print("=" * 60)
    print("RipeSense verification")
    print("=" * 60)

    check_imports()
    check_data()
    check_results()
    check_models()
    check_figures()
    if args.full:
        check_inference()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        print("\nFix failures, then re-run. If models/figures missing:")
        print("  py -m src.run_pipeline")
        sys.exit(1)
    print("\nAll checks passed.")
    if not args.full:
        print("Run with --full to test Live Prediction inference.")


if __name__ == "__main__":
    main()
