"""Phase 3 (part 2) - Knowledge-graph feature engineering (docs/03, docs/04).

Converts validated KG rules into tabular features following Perkovic et al. [8]:
  1) binary activation flags
  2) continuous risk scores (how far a reading is past its threshold)
  3) an aggregate violation count per row

Rules are validated on the TRAINING set only: a rule is kept if its activation
rate >= 5% AND a chi-squared test shows a significant association (p < 0.05)
with the ripeness label. The generator is fittable/serialisable so the same
thresholds are reused at inference time in the Streamlit app.

Viva tip: this is knowledge *integration* (extra columns), not a separate
expert system — the tree model still learns from data.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency  # Used for: rule–label association test (p < 0.05)

from . import config as C


def _resolve_threshold(series: pd.Series, q: float, fixed) -> float:
    """Used for: prefer published literature threshold when inside observed range, else quantile."""
    quantile_val = float(series.quantile(q))
    if fixed is not None and series.min() <= fixed <= series.max():
        return float(fixed)
    return quantile_val


def _condition(series: pd.Series, op: str, thr: float) -> pd.Series:
    """Used for: boolean mask — is sensor reading above/below rule threshold?"""
    return (series > thr) if op == ">" else (series < thr)


def _risk(series: pd.Series, op: str, thr: float) -> pd.Series:
    """Used for: continuous risk_R* feature — distance past threshold, capped at 5.0."""
    eps = 1e-9
    if op == ">":
        r = (series - thr) / (abs(thr) + eps)
    else:
        r = (thr - series) / (abs(thr) + eps)
    return r.clip(lower=0.0, upper=5.0)


class KGFeatureGenerator:
    """Used for: fit once on train (validate rules), transform at train/test/inference."""

    def __init__(self):
        self.rules: list[dict] = []          # accepted simple rules (with thr)
        self.interactions: list[dict] = []   # accepted interaction rules (R13–R15)
        self.validation_log: list[dict] = [] # shown on Knowledge Graph app page
        self.feature_names: list[str] = []   # e.g. flag_R1, risk_R1, kg_violation_count

    # ------------------------------------------------------------------ fit #
    def fit(self, X_train_raw: pd.DataFrame, y_train: pd.Series) -> "KGFeatureGenerator":
        """Used for: learn thresholds + chi-squared gate on training data only."""
        self.rules = []
        self.interactions = []
        self.validation_log = []

        # ---- simple single-feature rules ---- #
        for spec in C.KG_RULES:
            s = X_train_raw[spec["subject"]]
            thr = _resolve_threshold(s, spec["q"], spec.get("fixed"))
            cond = _condition(s, spec["op"], thr)
            entry = self._validate(spec["id"], cond, y_train, spec, thr)
            if entry["accepted"]:
                self.rules.append({**spec, "threshold": thr})

        # ---- interaction rules ---- #
        for spec in C.KG_INTERACTION_RULES:
            ls, rs = X_train_raw[spec["left"]], X_train_raw[spec["right"]]
            lthr = _resolve_threshold(ls, spec["left_q"], None)
            rthr = _resolve_threshold(rs, spec["right_q"], None)
            cond = _condition(ls, spec["left_op"], lthr) & _condition(rs, spec["right_op"], rthr)
            entry = self._validate(spec["id"], cond, y_train, spec, (lthr, rthr))
            if entry["accepted"]:
                self.interactions.append({**spec, "left_threshold": lthr, "right_threshold": rthr})

        # ---- record resulting feature schema ---- #
        self.feature_names = []
        for r in self.rules:
            self.feature_names += [f"flag_{r['id']}", f"risk_{r['id']}"]
        for r in self.interactions:
            self.feature_names += [f"flag_{r['id']}"]
        self.feature_names += ["kg_violation_count"]
        return self

    def _validate(self, rule_id, cond, y_train, spec, thr) -> dict:
        """Used for: accept rule if activation >= 5% AND chi-squared p < 0.05 vs label."""
        activation = float(cond.mean())
        accepted = activation >= C.MIN_ACTIVATION_RATE
        p_value = None
        reason = ""
        if not accepted:
            reason = f"activation {activation:.3f} < {C.MIN_ACTIVATION_RATE}"
        else:
            # Used for: chi-squared test — is rule firing associated with ripeness stage?
            table = pd.crosstab(cond, y_train)
            if table.shape[0] < 2:
                accepted = False
                reason = "no variation in activation"
            else:
                chi2, p_value, _, _ = chi2_contingency(table)
                p_value = float(p_value)
                accepted = p_value < C.CHI2_ALPHA
                reason = "accepted" if accepted else f"chi2 p={p_value:.4f} >= {C.CHI2_ALPHA}"
        entry = dict(
            rule_id=rule_id,
            activation_rate=round(activation, 4),
            p_value=p_value,
            accepted=bool(accepted),
            reason=reason,
            threshold=thr,
            predicate=spec.get("predicate"),
            object=spec.get("object"),
            expected_dir=spec.get("expected_dir"),
            source=spec.get("source"),
        )
        self.validation_log.append(entry)
        return entry

    # ------------------------------------------------------------ transform #
    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Used for: build flag_*, risk_*, kg_violation_count columns from raw sensors."""
        out = pd.DataFrame(index=X_raw.index)
        violation = pd.Series(0, index=X_raw.index, dtype=float)

        for r in self.rules:
            s = X_raw[r["subject"]]
            cond = _condition(s, r["op"], r["threshold"])
            out[f"flag_{r['id']}"] = cond.astype(int)
            out[f"risk_{r['id']}"] = _risk(s, r["op"], r["threshold"]).astype(float)
            violation = violation + cond.astype(float)

        for r in self.interactions:
            ls, rs = X_raw[r["left"]], X_raw[r["right"]]
            cond = (_condition(ls, r["left_op"], r["left_threshold"]) &
                    _condition(rs, r["right_op"], r["right_threshold"]))
            out[f"flag_{r['id']}"] = cond.astype(int)
            violation = violation + cond.astype(float)

        out["kg_violation_count"] = violation
        return out[self.feature_names]

    def fired_rules(self, row: dict) -> list[dict]:
        """Used for: Decision Support — list which literature rules fired for one reading."""
        fired = []
        for r in self.rules:
            val = row[r["subject"]]
            cond = (val > r["threshold"]) if r["op"] == ">" else (val < r["threshold"])
            if cond:
                fired.append({
                    "rule_id": r["id"],
                    "text": f"{r['subject']} {r['op']} {r['threshold']:.2f} "
                            f"-> {r['predicate']} {r['object']}",
                    "predicate": r["predicate"],
                    "object": r["object"],
                    "subject": r["subject"],
                    "op": r["op"],
                    "expected_dir": r["expected_dir"],
                    "source": r["source"],
                })
        for r in self.interactions:
            lc = (row[r["left"]] > r["left_threshold"]) if r["left_op"] == ">" else (row[r["left"]] < r["left_threshold"])
            rc = (row[r["right"]] > r["right_threshold"]) if r["right_op"] == ">" else (row[r["right"]] < r["right_threshold"])
            if lc and rc:
                fired.append({
                    "rule_id": r["id"],
                    "text": r["label"],
                    "predicate": r["predicate"],
                    "object": r["object"],
                    "expected_dir": r["expected_dir"],
                    "source": r["source"],
                })
        return fired

    # ------------------------------------------------------------ persist #
    def to_dict(self) -> dict:
        return {
            "rules": self.rules,
            "interactions": self.interactions,
            "feature_names": self.feature_names,
            "validation_log": self.validation_log,
        }

    def save(self, path: str) -> None:
        """Used for: kg_generator.json — same thresholds at app inference as in training."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "KGFeatureGenerator":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        obj = cls()
        obj.rules = d["rules"]
        obj.interactions = d["interactions"]
        obj.feature_names = d["feature_names"]
        obj.validation_log = d.get("validation_log", [])
        return obj


def build_augmented(generator: KGFeatureGenerator,
                    X_sensors_scaled: pd.DataFrame,
                    X_raw: pd.DataFrame) -> pd.DataFrame:
    """Used for: 6 scaled sensors + KG columns → 32-feature matrix for kg_rf / kg_xgb."""
    kg = generator.transform(X_raw)
    return pd.concat([X_sensors_scaled, kg], axis=1)
