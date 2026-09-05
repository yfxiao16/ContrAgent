"""Phase 4: score AgentPex's LLM-judge against the deterministic GT labels.

Reads manifest.json (sim_id -> GT categories) and AgentPex test_eval_*.json
files (per-trace category scores, 0-100, 100 = clean). Maps AgentPex score
keys to our categories, thresholds score < TAU as "flagged violation", and
reports precision/recall/F1 per category and for any-violation, restricted to
the categories BOTH systems evaluate (so AgentPex is not penalised for flagging
categories outside our deterministic GT).

Usage:
  /opt/homebrew/bin/python3.13 score_f1.py <artifact_dir> [<artifact_dir> ...]
  (artifact dirs are read from artifact_*.path if none given)
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# AgentPex score-key (without _score suffix) -> our GT category.
# Only categories present in the deterministic honest GT are compared.
KEY2CAT = {
    "output_spec_eval": "output_spec",
    "transition_spec_eval": "transition_spec",
    "arg_spec_eval": "argument_spec",
    "plan_eval": "predicted_plan",
    # forbidden_edges: resolved at runtime if a dedicated score key exists,
    # else folded into transition_spec (see FORBIDDEN_KEY below).
}
FORBIDDEN_KEY = "forbidden_edges_eval"  # may or may not exist in eval output
# categories that exist in our deterministic GT (universe of comparison)
GT_CATS = {"output_spec", "transition_spec", "argument_spec",
           "predicted_plan", "forbidden_edges"}


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p and r and (p + r) else 0.0
    return p, r, f


def load_eval_scores(artifact_dirs):
    """sim_id -> {category: score(0-100)}; also flag if forbidden score exists."""
    out = {}
    has_forbidden = False
    import os
    files = []
    for ad in artifact_dirs:
        files += glob.glob(f"{ad}/**/test_eval_*.json", recursive=True)
    # sort by mtime ascending so the most recent eval of a sim_id wins
    n_bad = 0
    for f in sorted(set(files), key=os.path.getmtime):
        try:
            d = json.load(open(f))
        except Exception:
            n_bad += 1
            continue
        if True:
            tid = d.get("trace_id", "")
            if not tid.startswith("tau_sq_sim_"):
                continue
            sim_id = tid[len("tau_sq_sim_"):]
            cats = {}
            for key, cat in KEY2CAT.items():
                v = d.get(f"{key}_score")
                if v is not None:
                    cats[cat] = float(v)
            fv = d.get(f"{FORBIDDEN_KEY}_score")
            if fv is not None:
                has_forbidden = True
                cats["forbidden_edges"] = float(fv)
            out[sim_id] = cats
    return out, has_forbidden


def main():
    artifact_dirs = sys.argv[1:]
    if not artifact_dirs:
        artifact_dirs = [Path(p).read_text().strip()
                         for p in glob.glob(str(HERE / "artifact_*.path"))]
    manifest = json.loads((HERE / "manifest.json").read_text())
    scores, has_forbidden = load_eval_scores(artifact_dirs)

    judged = [s for s in manifest if s in scores]
    print(f"GT sims: {len(manifest)} | AgentPex-judged: {len(judged)}")
    print(f"forbidden_edges has its own eval score: {has_forbidden}")
    if not judged:
        print("No overlap between manifest and eval outputs.")
        return

    # the categories AgentPex actually emits a score for, intersected w/ GT
    emitted = set()
    for s in judged:
        emitted |= set(scores[s])
    cmp_cats = sorted(emitted & GT_CATS)
    print(f"compared categories: {cmp_cats}\n")

    for tau in (50, 60, 65, 70, 80, 90, 99):
        # per-category
        print(f"--- TAU = {tau} (score < TAU => AgentPex flags violation) ---")
        # any-violation over compared categories
        any_tp = any_fp = any_fn = any_tn = 0
        per_cat = {}
        for cat in cmp_cats:
            tp = fp = fn = 0
            for s in judged:
                if cat not in scores[s]:
                    continue
                gt = cat in manifest[s]["gt_categories"]
                pred = scores[s][cat] < tau
                tp += gt and pred
                fp += (not gt) and pred
                fn += gt and (not pred)
            per_cat[cat] = _prf(tp, fp, fn)
        for s in judged:
            gt_any = any(c in manifest[s]["gt_categories"] for c in cmp_cats)
            pred_any = any(scores[s].get(c, 100) < tau for c in cmp_cats)
            any_tp += gt_any and pred_any
            any_fp += (not gt_any) and pred_any
            any_fn += gt_any and (not pred_any)
            any_tn += (not gt_any) and (not pred_any)
        for cat in cmp_cats:
            p, r, f = per_cat[cat]
            gtpos = sum(1 for s in judged if cat in manifest[s]["gt_categories"])
            print(f"  {cat:16} P={p:.2f} R={r:.2f} F1={f:.2f}  (GT pos={gtpos})")
        ap, ar, af = _prf(any_tp, any_fp, any_fn)
        acc = (any_tp + any_tn) / len(judged)
        print(f"  {'ANY-VIOLATION':16} P={ap:.2f} R={ar:.2f} F1={af:.2f}  "
              f"acc={acc:.2f}  (TP{any_tp} FP{any_fp} FN{any_fn} TN{any_tn})\n")


if __name__ == "__main__":
    main()
