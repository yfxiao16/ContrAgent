#!/usr/bin/env python3
"""Score ContrAgent on EXACTLY the instances the LLM judge saw.

Reads sample_manifest.json (written by llm_judge.py), copies those trace files
into a temp dir per domain, runs `contragent eval --json` with the same
hand-authored contracts, and reports per-domain + aggregate recall/FPR/precision/F1
at the trace level (unsafe=positive, blocked-by->=1-contract = predicted unsafe).

This makes the head-to-head strictly paired on identical files.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
TRACES = HERE / "traces"
CONTRACTS = HERE / "contracts"
SAMPLE_PATH = HERE / "sample_manifest.json"
OUT_PATH = HERE / "contragent_paired_results.json"


def f1_from(recall, precision):
    if precision is None or (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    if not SAMPLE_PATH.exists():
        sys.exit("sample_manifest.json not found; run llm_judge.py first.")
    manifest = json.loads(SAMPLE_PATH.read_text())

    per_domain = {}
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for domain, files in manifest.items():
            cfg = CONTRACTS / f"{domain}.yaml"
            if not cfg.exists():
                print(f"[{domain}] no contract; skipping", file=sys.stderr)
                continue
            ddir = tmp / domain
            ddir.mkdir(parents=True)
            for name in files:
                src = TRACES / domain / name
                shutil.copy(src, ddir / name)

            cmd = [
                sys.executable,
                "-m",
                "contragent.cli",
                "eval",
                str(ddir),
                "--config",
                str(cfg),
                "--agent",
                "*",
                "--json",
            ]
            res = subprocess.run(
                cmd,
                cwd=str(REPO),
                capture_output=True,
                text=True,
                env={**__import__("os").environ, "PYTHONPATH": "."},
            )
            try:
                data = json.loads(res.stdout)
            except json.JSONDecodeError:
                print(f"[{domain}] eval failed:\n{res.stderr[-500:]}", file=sys.stderr)
                continue
            o = data["overall"]
            tp, fp, fn, tn = o["tp"], o["fp"], o["fn"], o["tn"]
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            precision = tp / (tp + fp) if (tp + fp) else None
            per_domain[domain] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "recall": round(100 * recall, 1),
                "fpr": round(100 * fpr, 1),
                "precision": round(100 * precision, 1) if precision is not None else None,
                "f1": round(100 * f1_from(recall, precision or 0.0), 1),
                "n": data["n_cases"],
            }
            for k in agg:
                agg[k] += o[k]

    tp, fp, fn, tn = agg["tp"], agg["fp"], agg["fn"], agg["tn"]
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else None
    aggregate = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": round(100 * recall, 1),
        "fpr": round(100 * fpr, 1),
        "precision": round(100 * precision, 1) if precision is not None else None,
        "f1": round(100 * f1_from(recall, precision or 0.0), 1),
        "n": tp + fp + fn + tn,
    }

    out = {"per_domain": per_domain, "aggregate": aggregate}
    OUT_PATH.write_text(json.dumps(out, indent=2))

    print("\n=== CONTRAGENT (paired, same sampled files) ===")
    print(f"{'domain':14} {'recall':>7} {'fpr':>6} {'prec':>6} {'f1':>6}  n")
    for domain, m in per_domain.items():
        print(
            f"{domain:14} {m['recall']:>7} {m['fpr']:>6} "
            f"{str(m['precision']):>6} {m['f1']:>6}  {m['n']}"
        )
    a = aggregate
    print(
        f"{'AGG':14} {a['recall']:>7} {a['fpr']:>6} {str(a['precision']):>6} {a['f1']:>6}  {a['n']}"
    )


if __name__ == "__main__":
    main()
