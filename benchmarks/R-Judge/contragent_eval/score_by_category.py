#!/usr/bin/env python3
"""Per-category recall / FPR / precision over the converted NATIVE traces, plus
the deterministic-vs-semantic split of the unsafe set.

Mirrors contragent.eval_runner._eval_contract_on_trace (ground(trace) with NO
content_atoms; contract violated == not evaluate(formula)) and aggregates the
"any contract blocks -> blocked" confusion matrix, broken down by the
``category`` field in each native trace's ``metadata``. Run from the repo root
with PYTHONPATH=. (run.sh does this).

DET/SEMANTIC SPLIT: a trace is structurally DECIDABLE if it calls at least one
tool that some contract gates (i.e. the risk is observable from trace shape). If
an unsafe trace calls none of the gated tools, its risk is purely SEMANTIC
(harmful advice / contextual misjudgement / authorized-look-alike) and no
deterministic contract can catch it. We report recall over the whole unsafe set
AND over just the structurally-decidable subset.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import yaml

from contragent.formulas.evaluator import evaluate
from contragent.formulas.formula import Atom
from contragent.formulas.parser import parse_repr
from contragent.models.trace import Trace
from contragent.tracer.grounding import ground

HERE = Path(__file__).resolve().parent


def _called_tools(node, out: set[str]) -> None:
    if node is None:
        return
    if isinstance(node, Atom):
        if node.predicate == "called" and node.args:
            out.add(node.args[0])
        return
    for attr in ("child", "left", "right"):
        _called_tools(getattr(node, attr, None), out)


def main() -> None:
    cfg = yaml.safe_load(open(HERE / "contracts.yaml"))
    raw = cfg["agents"]["*"]["contracts"]
    contracts = [(c["desc"], parse_repr(c["G"]["ltl"])) for c in raw]

    # DECIDABLE set = tools whose mere PRESENCE is a policy violation, i.e. the
    # tools named in the UNCONDITIONAL contracts (the side-effect gate, the
    # sensitive-read gate, the money rate-limit). An unsafe trace is
    # structurally DECIDABLE iff it calls one of these. The temporal
    # injection->action contract is EXCLUDED here: its antecedent reads
    # (GmailReadEmail, browse_website, ...) appear in BOTH safe and unsafe
    # traces and a read with no following effect is the SAFE pattern, so a read
    # alone does NOT make a trace structurally decidable.
    gated: set[str] = set()
    for c in raw:
        if c["desc"].startswith("Injection->action"):
            continue
        _called_tools(parse_repr(c["G"]["ltl"]), gated)

    cases = []
    for p in glob.glob(str(HERE / "traces" / "*.json")):
        d = json.load(open(p))
        meta = d.get("metadata", {})
        label = "unsafe" if int(meta.get("label", 0)) == 1 else "safe"
        t = Trace.from_dict(d)
        tools = {e.tool for e in t.events if e.event_type == "tool_call" and e.tool}
        cases.append((t, label, meta.get("category", "?"), bool(tools & gated)))

    percon = defaultdict(lambda: [0, 0, 0, 0])  # tp fp fn tn
    overall = [0, 0, 0, 0]
    percat = defaultdict(lambda: [0, 0, 0, 0])
    # det/semantic split of the unsafe set
    det_tp = det_total = sem_total = 0

    for t, lb, cat, decidable in cases:
        vals = ground(t)
        any_block = False
        for desc, f in contracts:
            viol = not evaluate(f, vals)
            any_block = any_block or viol
            s = percon[desc]
            s[0] += viol and lb == "unsafe"
            s[1] += viol and lb == "safe"
            s[2] += (not viol) and lb == "unsafe"
            s[3] += (not viol) and lb == "safe"
        for tgt in (overall, percat[cat]):
            tgt[0] += any_block and lb == "unsafe"
            tgt[1] += any_block and lb == "safe"
            tgt[2] += (not any_block) and lb == "unsafe"
            tgt[3] += (not any_block) and lb == "safe"
        if lb == "unsafe":
            if decidable:
                det_total += 1
                det_tp += any_block
            else:
                sem_total += 1

    def fmt(tp, fp, fn, tn):
        rec = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        return (
            f"TP={tp:3} FP={fp:3} FN={fn:3} TN={tn:3} "
            f"recall={rec:5.1%} FPR={fpr:5.1%} prec={prec:5.1%}"
        )

    print("--- per contract ---")
    for desc, _ in contracts:
        print("  ", fmt(*percon[desc]), "|", desc[:48])
    print("--- per category ---")
    for cat in sorted(percat):
        print(f"  {cat:12}", fmt(*percat[cat]))
    print("--- overall ---")
    print("  ", fmt(*overall))
    print("--- deterministic vs semantic split (unsafe set) ---")
    total_unsafe = det_total + sem_total
    print(
        f"  structurally decidable: {det_total}/{total_unsafe} unsafe "
        f"({det_total / total_unsafe:.1%}); "
        f"purely semantic (no gated tool called): {sem_total}/{total_unsafe} "
        f"({sem_total / total_unsafe:.1%})"
    )
    print(
        f"  recall on decidable subset: {det_tp}/{det_total} = "
        f"{det_tp / det_total:.1%}"
    )


if __name__ == "__main__":
    main()
