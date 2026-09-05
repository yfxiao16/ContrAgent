#!/usr/bin/env python3
"""Eval harness that MIRRORS contragent.eval_runner._eval_contract_on_trace
(grounds with collect_content_atoms, so content predicates fire), then reports
overall + per-category + per-contract confusion matrices and the FP/FN lists.

Use: PYTHONPATH=. python benchmarks/R-Judge/contragent_eval/_harness.py [contracts.yaml]
"""
from __future__ import annotations
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import yaml

from contragent.formulas.evaluator import evaluate
from contragent.formulas.parser import parse_repr
from contragent.tracer.grounding import ground, collect_content_atoms
from contragent.models.trace import Trace

HERE = Path(__file__).resolve().parent


def load_contracts(path):
    cfg = yaml.safe_load(open(path))
    raw = cfg["agents"]["*"]["contracts"]
    out = []
    for c in raw:
        f = parse_repr(c["G"]["ltl"])
        ca = collect_content_atoms([f]) or None
        out.append((c["desc"], f, ca))
    return out


def main():
    cpath = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "contracts.yaml")
    contracts = load_contracts(cpath)

    overall = [0, 0, 0, 0]  # tp fp fn tn
    percat = defaultdict(lambda: [0, 0, 0, 0])
    percon = defaultdict(lambda: [0, 0, 0, 0])
    fps, fns = [], []

    for p in sorted(glob.glob(str(HERE / "traces" / "*.json"))):
        d = json.load(open(p))
        meta = d.get("metadata", {})
        lab = "unsafe" if int(meta.get("label", 0)) == 1 else "safe"
        cat = meta.get("category", "?")
        t = Trace.from_dict(d)
        any_block = False
        for desc, f, ca in contracts:
            vals = ground(t, content_atoms=ca)
            viol = not evaluate(f, vals)
            any_block = any_block or viol
            s = percon[desc]
            s[0] += viol and lab == "unsafe"
            s[1] += viol and lab == "safe"
            s[2] += (not viol) and lab == "unsafe"
            s[3] += (not viol) and lab == "safe"
        for tgt in (overall, percat[cat]):
            tgt[0] += any_block and lab == "unsafe"
            tgt[1] += any_block and lab == "safe"
            tgt[2] += (not any_block) and lab == "unsafe"
            tgt[3] += (not any_block) and lab == "safe"
        if any_block and lab == "safe":
            fps.append((cat, Path(p).stem))
        if (not any_block) and lab == "unsafe":
            fns.append((cat, Path(p).stem))

    def fmt(tp, fp, fn, tn):
        rec = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return f"TP={tp:3} FP={fp:3} FN={fn:3} TN={tn:3} rec={rec:5.1%} FPR={fpr:5.1%} prec={prec:5.1%} F1={f1*100:4.1f}"

    print("--- per contract ---")
    for desc, _, _ in contracts:
        print("  ", fmt(*percon[desc]), "|", desc[:46])
    print("--- per category ---")
    for cat in sorted(percat):
        print(f"  {cat:12}", fmt(*percat[cat]))
    print("--- overall ---")
    print("  ", fmt(*overall))
    print(f"--- {len(fps)} FP ---")
    for cat, n in fps:
        print("  FP", cat, n)
    print(f"--- {len(fns)} FN (first 50) ---")
    for cat, n in fns[:50]:
        print("  FN", cat, n)


if __name__ == "__main__":
    main()
