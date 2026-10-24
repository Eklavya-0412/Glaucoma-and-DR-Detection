"""Module N - transparent, configurable, *experimental* screening flags.

These are NOT diagnoses. Each rule is a simple comparison on measured
features, loaded from ``screening_rules.json`` so it can be inspected and
changed. Per-lesion count thresholds for the DR rule were chosen (Youden
index) on the STARE tuning split (im0001-im0100) and the rule is evaluated
on the held-out split (im0101-im0402) by ``scripts/evaluate_stare.py``.
No validated clinical thresholds are claimed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RULES_PATH = Path(__file__).with_name("screening_rules.json")


def load_rules(path: Path = RULES_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def _compare(value, op: str, thr) -> bool | None:
    if value is None:
        return None
    return {">=": value >= thr, ">": value > thr, "<=": value <= thr, "<": value < thr}[op]


def evaluate_rules(features: dict[str, Any], rules: dict[str, Any] | None = None) -> dict[str, Any]:
    rules = rules or load_rules()
    out = {}
    for name, rule in rules["flags"].items():
        conds = []
        for c in rule["conditions"]:
            v = features.get(c["feature"])
            conds.append({**c, "value": v, "met": _compare(v, c["op"], c["threshold"])})
        mets = [c["met"] for c in conds]
        if rule["combine"] == "any":
            flag = True if any(m is True for m in mets) else (None if any(m is None for m in mets) else False)
        elif rule["combine"] == "at_least":
            k = int(rule["min_count"])
            n_true, n_none = sum(m is True for m in mets), sum(m is None for m in mets)
            flag = True if n_true >= k else (None if n_true + n_none >= k else False)
        else:
            flag = None if any(m is None for m in mets) else all(mets)
        out[name] = {
            "label": rule["label"], "flag": flag,
            "result": ("insufficient data" if flag is None else
                       rule["positive_text"] if flag else rule["negative_text"]),
            "combine": rule["combine"] if rule["combine"] != "at_least" else f"at least {rule['min_count']}",
            "conditions": conds, "basis": rule["basis"], "evaluation": rule.get("evaluation"),
        }
    return {"flags": out, "disclaimer": rules["disclaimer"]}
