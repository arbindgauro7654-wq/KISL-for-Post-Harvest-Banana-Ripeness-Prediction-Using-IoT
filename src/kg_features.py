"""Phase 3 (part 2) - Knowledge-graph feature engineering (docs/03, docs/04).

Converts validated KG rules into tabular features following Perkovic et al. [8]:
  1) binary activation flags
  2) continuous risk scores (how far a reading is past its threshold)
  3) an aggregate violation count per row

Rules are validated on the TRAINING set only: a rule is kept if its activation
rate >= 5% AND a chi-squared test shows a significant association (p < 0.05)
with the ripeness label. The generator is fittable/serialisable so the same
thresholds are reused at inference time in the Streamlit app.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from . import config as C


def _resolve_threshold(series: pd.Series, q: float, fixed) -> float:
    """Published threshold preferred when within observed range, else quantile."""
    quantile_val = float(series.quantile(q))
    if fixed is not None and series.min() <= fixed <= series.max():
        return float(fixed)
    return quantile_val


def _condition(series: pd.Series, op: str, thr: float) -> pd.Series:
    return (series > thr) if op == ">" else (series < thr)


def _risk(series: pd.Series, op: str, thr: float) -> pd.Series:
    """How far past the boundary (>=0), capped for stability."""
    eps = 1e-9
    if op == ">":
        r = (series - thr) / (abs(thr) + eps)
    else:
        r = (thr - series) / (abs(thr) + eps)
    return r.clip(lower=0.0, upper=5.0)


class KGFeatureGenerator:
    """Fit thresholds + validate rules on train; transform any sensor frame."""

    def __init__(self):
        self.rules: list[dict] = []          # accepted simple rules (with thr)
        self.interactions: list[dict] = []   # accepted interaction rules
        self.validation_log: list[dict] = []
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------ fit #
    def fit(self, X_train_raw: pd.DataFrame, y_train: pd.Series) -> "KGFeatureGenerator":
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
        activation = float(cond.mean())
        accepted = activation >= C.MIN_ACTIVATION_RATE
        p_value = None
        reason = ""
        if not accepted:
            reason = f"activation {activation:.3f} < {C.MIN_ACTIVATION_RATE}"
        else:
            # chi-squared association between rule activation and label
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
        """Return the list of rules that fire for a single raw sensor reading.

        Used by the decision-support tool and the app to explain a prediction.
        """
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
    """Concatenate the six scaled sensors with KG features (computed from raw)."""
    kg = generator.transform(X_raw)
    return pd.concat([X_sensors_scaled, kg], axis=1)
