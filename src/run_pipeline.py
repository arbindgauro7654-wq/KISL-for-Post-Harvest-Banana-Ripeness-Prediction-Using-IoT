"""End-to-end pipeline orchestration (docs/03, Phases 1-6).

Run with:  py -m src.run_pipeline
Produces all artefacts under outputs/ that the Streamlit app consumes.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import pandas as pd

from . import config as C
from . import knowledge_graph as kgmod
from . import eda as edamod
from . import evaluate as ev
from . import train as trainmod
from .data_loader import load_data, feature_summary
from .decision_support import predict_one
from .kg_features import KGFeatureGenerator, build_augmented
from .robustness import run_robustness


def _save_json(obj, name):
    with open(os.path.join(C.RESULT_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    t0 = time.time()
    print("=" * 70)
    print("Knowledge-Integrated Banana Ripeness Pipeline")
    print("=" * 70)

    # ----- Phase 1: data ------------------------------------------------- #
    print("[Phase 1] Loading & preprocessing data ...")
    data = load_data()
    _save_json(data.report, "data_report.json")
    joblib.dump(data.scaler, os.path.join(C.MODEL_DIR, "scaler.pkl"))
    print(f"  train={data.report['n_train']} test={data.report['n_test']} "
          f"smote_applied={data.report['smote_applied']}")

    # ----- Phase 2: EDA -------------------------------------------------- #
    print("[Phase 2] Exploratory data analysis ...")
    eda_summary = edamod.run_eda(data.X_train_raw, data.y_train)

    # ----- Phase 3: KG + features --------------------------------------- #
    print("[Phase 3] Building knowledge graph & features ...")
    G = kgmod.build_graph()
    kgmod.save_graph(G)
    kg_summary_text = kgmod.summary(G)
    with open(os.path.join(C.RESULT_DIR, "kg_summary.txt"), "w", encoding="utf-8") as f:
        f.write(kg_summary_text)

    generator = KGFeatureGenerator().fit(data.X_train_raw, data.y_train)
    generator.save(os.path.join(C.MODEL_DIR, "kg_generator.json"))
    _save_json(generator.validation_log, "validated_rules.json")
    n_accept = sum(1 for r in generator.validation_log if r["accepted"])
    print(f"  KG rules accepted: {n_accept}/{len(generator.validation_log)} "
          f"-> {len(generator.feature_names)} features")

    X_train_aug = build_augmented(generator, data.X_train_scaled, data.X_train_raw)
    X_test_aug = build_augmented(generator, data.X_test_scaled, data.X_test_raw)

    # ----- Phase 4: training -------------------------------------------- #
    print("[Phase 4] Training & tuning 4 models (5-fold CV) ...")
    trained = trainmod.train_all(data.X_train_scaled, X_train_aug, data.y_train)
    models = trained["models"]

    feat_map = {
        "baseline_rf": (data.X_train_scaled, data.X_test_scaled, False),
        "kg_rf": (X_train_aug, X_test_aug, True),
        "baseline_xgb": (data.X_train_scaled, data.X_test_scaled, False),
        "kg_xgb": (X_train_aug, X_test_aug, True),
    }

    # ----- Phase 5: evaluation ------------------------------------------ #
    print("[Phase 5] Evaluation (RQ1 metrics, confusion, comparison) ...")
    results = {}
    for name, model in models.items():
        Xtr, Xte, is_kg = feat_map[name]
        results[name] = {
            "is_kg": is_kg,
            "cv": trained["cv_meta"][name],
            "train": ev.classification_metrics(model, Xtr, data.y_train),
            "test": ev.classification_metrics(model, Xte, data.y_test),
        }
        ev.confusion_fig(model, Xte, data.y_test,
                         f"Confusion matrix - {name}", f"confusion_{name}.png")

    comp_df = ev.comparison_fig(results)

    # RQ1 significance: baseline vs KG within each algorithm
    mcnemar = {
        "rf_baseline_vs_kg": ev.mcnemar_test(
            models["baseline_rf"], data.X_test_scaled,
            models["kg_rf"], X_test_aug, data.y_test),
        "xgb_baseline_vs_kg": ev.mcnemar_test(
            models["baseline_xgb"], data.X_test_scaled,
            models["kg_xgb"], X_test_aug, data.y_test),
    }

    # ----- RQ2: SHAP importance + alignment ----------------------------- #
    print("[Phase 5] SHAP interpretability & rule alignment (RQ2) ...")
    sample = X_test_aug.sample(min(C.SHAP_SAMPLE_SIZE, len(X_test_aug)),
                               random_state=C.RANDOM_SEED)
    sample_base = data.X_test_scaled.loc[sample.index]

    shap_kg = ev.shap_importance(models["kg_rf"], sample, "kg_rf")
    shap_base = ev.shap_importance(models["baseline_rf"], sample_base, "baseline_rf")
    alignment = ev.rule_alignment_score(generator, shap_kg["importance"])
    _save_json({"kg_rf": alignment}, "rq2_alignment.json")

    # ----- RQ3: robustness ---------------------------------------------- #
    print("[Phase 5] Robustness stress-tests (RQ3) ...")
    train_ranges = {c: float(data.X_train_raw[c].max() - data.X_train_raw[c].min())
                    for c in C.SENSOR_FEATURES}
    train_medians = {c: float(data.X_train_raw[c].median()) for c in C.SENSOR_FEATURES}

    # best KG model vs its baseline counterpart
    best_name = max(results, key=lambda n: results[n]["test"]["macro_f1"])
    robustness = {
        "kg_rf": run_robustness(models["kg_rf"], data.scaler, generator,
                                data.X_test_raw, data.y_test, True,
                                train_ranges, train_medians),
        "baseline_rf": run_robustness(models["baseline_rf"], data.scaler, generator,
                                      data.X_test_raw, data.y_test, False,
                                      train_ranges, train_medians),
    }
    _save_json(robustness, "robustness.json")
    _robustness_fig(robustness)

    # ----- Phase 6: decision-support demo ------------------------------- #
    print("[Phase 6] Decision-support demo ...")
    best_kg_model = models["kg_rf"]
    demo_rows = data.X_test_raw.sample(5, random_state=C.RANDOM_SEED)
    demo = []
    for idx, row in demo_rows.iterrows():
        out = predict_one(best_kg_model, data.scaler, generator,
                          row.to_dict(), is_kg=True)
        out["true_stage"] = int(data.y_test.loc[idx])
        demo.append(out)
    _save_json(demo, "decision_support_demo.json")

    # ----- assemble final results --------------------------------------- #
    best_model_name = "kg_rf" if results["kg_rf"]["test"]["macro_f1"] >= results["kg_xgb"]["test"]["macro_f1"] else "kg_xgb"
    joblib.dump(models[best_model_name], os.path.join(C.MODEL_DIR, "best_model.pkl"))

    final = {
        "data_report": data.report,
        "feature_summary": feature_summary(data.X_train_raw).to_dict(),
        "models": results,
        "mcnemar": mcnemar,
        "rq2_alignment": {"kg_rf": alignment},
        "robustness": robustness,
        "best_model": {"name": best_model_name, "is_kg": results[best_model_name]["is_kg"]},
        "kg": {"n_rules_total": len(generator.validation_log),
               "n_rules_accepted": n_accept,
               "feature_names": generator.feature_names},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    _save_json(final, "model_results.json")
    _write_text_report(final)

    print("-" * 70)
    print(f"DONE in {final['runtime_seconds']}s. Best model: {best_model_name}")
    for n, r in results.items():
        print(f"  {n:14s} test macro-F1={r['test']['macro_f1']:.4f} acc={r['test']['accuracy']:.4f}")
    print("-" * 70)


def _robustness_fig(robustness):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, kind, title in zip(axes, ("noise", "missing"),
                               ("Gaussian noise", "Missing values")):
        for model_name, res in robustness.items():
            levels = list(res[kind].keys())
            pcts = [res[kind][l]["pct_of_clean"] for l in levels]
            ax.plot(levels, pcts, marker="o", label=model_name)
        ax.axhline(80, ls="--", color="red", label="80% threshold")
        ax.set_title(f"Robustness to {title}")
        ax.set_xlabel("Degradation level")
        ax.set_ylabel("Macro-F1 (% of clean)")
        ax.set_ylim(0, 105)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "robustness_curves.png"), dpi=130)
    plt.close(fig)


def _write_text_report(final):
    lines = ["KNOWLEDGE-INTEGRATED BANANA RIPENESS - RESULTS", "=" * 50, ""]
    lines.append(f"Train rows: {final['data_report']['n_train']}  "
                 f"Test rows: {final['data_report']['n_test']}")
    lines.append(f"SMOTE applied: {final['data_report']['smote_applied']} "
                 f"(min class fraction={final['data_report']['min_class_fraction']:.3f})")
    lines.append(f"KG rules accepted: {final['kg']['n_rules_accepted']}/"
                 f"{final['kg']['n_rules_total']}")
    lines.append("")
    lines.append("MODEL PERFORMANCE (test set)")
    lines.append("-" * 50)
    header = f"{'model':16s}{'acc':>8s}{'macroF1':>10s}{'w-prec':>9s}{'w-rec':>9s}"
    lines.append(header)
    for n, r in final["models"].items():
        t = r["test"]
        lines.append(f"{n:16s}{t['accuracy']:>8.4f}{t['macro_f1']:>10.4f}"
                     f"{t['weighted_precision']:>9.4f}{t['weighted_recall']:>9.4f}")
    lines.append("")
    lines.append("RQ1 - McNemar (baseline vs KG)")
    for k, v in final["mcnemar"].items():
        lines.append(f"  {k}: p={v['p_value']} significant={v['significant']}")
    lines.append("")
    lines.append("RQ2 - SHAP rule alignment (kg_rf): "
                 f"{final['rq2_alignment']['kg_rf']['alignment_score']}")
    lines.append("")
    lines.append("RQ3 - Robustness (kg_rf, % of clean macro-F1)")
    rob = final["robustness"]["kg_rf"]
    for lvl, v in rob["noise"].items():
        lines.append(f"  noise {lvl}: {v['pct_of_clean']}%")
    for lvl, v in rob["missing"].items():
        lines.append(f"  missing {lvl}: {v['pct_of_clean']}%")
    lines.append(f"  sensor failure: {rob['sensor_failure']['pct_of_clean']}%")
    lines.append("")
    lines.append(f"Best model: {final['best_model']['name']}")
    lines.append(f"Runtime: {final['runtime_seconds']}s")

    with open(os.path.join(C.RESULT_DIR, "model_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

