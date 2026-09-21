"""Appendix experiments on the SOPBench RECORDED offline traces (0 LLM calls):

(A) Per-model violation profiles -- for each base model, the deterministic
    contract library's detection recall on its unsafe traces, its false-positive
    rate on safe traces, and which contract FAMILY fires most. Surfaces
    model-specific "willful disobedience."

(B) Oracle-grounding ablation (honest form). SOPBench's relevant atoms are
    STRUCTURAL (a tool was called; a call returned success), read directly off
    the trace, so grounding is exact. We verify the residual detection gap is
    contract COVERAGE, not AP grounding, by an oracle that perfects the single
    most load-bearing AP -- the goal action's success flag -- and measuring the
    recall delta (expected ~0).

Sampled (cap per domain x model) for tractable runtime.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
warnings.simplefilter("ignore")

DOMAINS = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]
CAP = 60  # per (domain, model, label) sample cap


def load_det_contracts(yaml_path):
    from contragent.config import load_config
    from contragent.eval_runner import resolve_entry

    cfg = load_config(yaml_path)
    agent = cfg.agents.get("*") or next(iter(cfg.agents.values()))
    out = []
    for ce in agent.contracts:
        for part in (ce.assumption, ce.guarantee):
            if part is None:
                continue
            for e in part if isinstance(part, list) else [part]:
                _nl, p = resolve_entry(e)
                if p is not None:
                    out.append((p, getattr(p, "desc", "") or _nl or ""))
    return out


def family(desc: str) -> str:
    d = desc.lower()
    if "must precede" in d or "before" in d or "order" in d:
        return "ordering"
    if "exist" in d:
        return "existence"
    if (
        "credit" in d
        or ">=" in d
        or "<=" in d
        or "threshold" in d
        or "score" in d
        or "balance" in d
        or "amount" in d
    ):
        return "value/threshold"
    if "login" in d or "auth" in d or "logged" in d:
        return "auth"
    if "gate" in d:
        return "gate"
    return "other"


def fires(parsed_list, trace_dict):
    """Return set of (desc) that fire (violated) on the trace."""
    from contragent.eval_runner import _eval_contract_on_trace
    from contragent.models.trace import Trace

    t = Trace.from_dict(trace_dict)
    hit = set()
    for p, desc in parsed_list:
        try:
            if _eval_contract_on_trace(p, t) is True:
                hit.add(desc)
        except Exception:
            continue
    return hit


def force_goal_success(trace_dict, goal):
    """Oracle: set succeeded=1 on every goal tool_call (perfect success AP)."""
    import copy

    d = copy.deepcopy(trace_dict)
    for e in d.get("events", []):
        if e.get("type") == "tool_call" and e.get("tool") == goal:
            e.setdefault("args", {})["succeeded"] = 1
    return d


def main():
    # per-model aggregates
    pm = collections.defaultdict(
        lambda: {"u": 0, "u_hit": 0, "s": 0, "s_fp": 0, "fam": collections.Counter()}
    )
    # oracle ablation pooled
    orc = {"u": 0, "ship": 0, "oracle": 0}

    for dom in DOMAINS:
        ydir = os.path.join(REPO, "contragent", "contracts", "sopbench", f"{dom}.yaml")
        if not os.path.isfile(ydir):
            continue
        parsed = load_det_contracts(ydir)
        # group files by (model, label)
        buckets = collections.defaultdict(list)
        for f in glob.glob(os.path.join(HERE, "traces", dom, "*.json")):
            base = os.path.basename(f)
            label = "unsafe" if base.startswith("unsafe") else "safe"
            buckets[label].append(f)
        # sample within each label by model via metadata
        for label, files in buckets.items():
            per_model_count = collections.Counter()
            for f in files:
                try:
                    d = json.load(open(f))
                except (ValueError, OSError):
                    continue
                md = d.get("metadata", {})
                model = md.get("model", "?")
                if per_model_count[model] >= CAP:
                    continue
                per_model_count[model] += 1
                hit = fires(parsed, d)
                rec = pm[model]
                if label == "unsafe":
                    rec["u"] += 1
                    if hit:
                        rec["u_hit"] += 1
                        for h in hit:
                            rec["fam"][family(h)] += 1
                    # oracle ablation (unsafe only)
                    goal = md.get("user_goal")
                    orc["u"] += 1
                    if hit:
                        orc["ship"] += 1
                    oh = fires(parsed, force_goal_success(d, goal))
                    if oh:
                        orc["oracle"] += 1
                else:
                    rec["s"] += 1
                    if hit:
                        rec["s_fp"] += 1
        print(f"  [{dom}] done", file=sys.stderr)

    # ---- report (A) per-model profiles ----
    print("\n=== (A) Per-model violation profiles (SOPBench offline, det, 0 LLM) ===")
    print(
        f"{'model':38s} {'recall':>7s} {'FPR':>6s} {'n_uns':>6s} {'n_safe':>6s}  top-family(share)"
    )
    rows = []
    for model, r in pm.items():
        rec = 100 * r["u_hit"] / r["u"] if r["u"] else float("nan")
        fpr = 100 * r["s_fp"] / r["s"] if r["s"] else float("nan")
        topfam = r["fam"].most_common(1)
        tf = (
            f"{topfam[0][0]} ({100 * topfam[0][1] / sum(r['fam'].values()):.0f}%)"
            if topfam
            else "-"
        )
        rows.append((rec, model, fpr, r["u"], r["s"], tf))
    for rec, model, fpr, nu, ns, tf in sorted(rows, key=lambda x: -x[0]):
        print(f"{model:38s} {rec:6.1f}% {fpr:5.1f}% {nu:6d} {ns:6d}  {tf}")

    # ---- report (B) oracle ablation ----
    print("\n=== (B) Oracle-grounding ablation (success-AP), pooled unsafe ===")
    sr = 100 * orc["ship"] / orc["u"] if orc["u"] else 0
    oc = 100 * orc["oracle"] / orc["u"] if orc["u"] else 0
    print(f"  n_unsafe={orc['u']}")
    print(f"  shipped recall : {sr:.1f}%")
    print(f"  oracle  recall : {oc:.1f}%  (force goal succeeded=1)")
    print(
        f"  delta          : {oc - sr:+.1f} pp  -> grounding is{' NOT' if abs(oc - sr) < 3 else ''} the bottleneck; "
        f"residual is {'coverage' if abs(oc - sr) < 3 else 'mixed'}"
    )

    json.dump(
        {"per_model": {m: pm[m] | {"fam": dict(pm[m]["fam"])} for m in pm}, "oracle": orc},
        open(os.path.join(HERE, "appendix_profiles_results.json"), "w"),
        indent=2,
        default=str,
    )


if __name__ == "__main__":
    main()
