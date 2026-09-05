"""Aggregate the SOPBench live sweep across seeds and models.

(1) flash 3-seed mean +/- std per domain & condition (success/safety) -> CI for
    the main table.
(2) gemini-2.5-pro single-seed table -> model-agnostic check.

Reads results/final_<d>.json (seed1), s2_<d>.json, s3_<d>.json (flash),
and pro_<d>.json (pro).
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DOMAINS = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]
CONDS = ["base", "prompt", "enforce", "llm_guard"]


def metrics(results, cond):
    rs = [r for r in results if r.get("condition") == cond and "error" not in r]
    pos = [r for r in rs if r["should_succeed"]]
    neg = [r for r in rs if not r["should_succeed"]]
    s = 100.0 * sum(1 for r in pos if r["goal_completed"]) / len(pos) if pos else None
    sf = 100.0 * sum(1 for r in neg if not r["goal_completed"]) / len(neg) if neg else None
    gc = sum(r.get("guard_calls", 0) for r in rs) / len(rs) if rs else 0
    return s, sf, gc


def load(prefix, dom):
    # seed1 uses the 'final_' prefix; others use their own
    p = os.path.join(RES, f"{prefix}_{dom}.json")
    return json.load(open(p))["results"] if os.path.isfile(p) else None


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    return m, sd


def main():
    # ---- (1) flash 3-seed ----
    seeds = ["final", "s2", "s3"]
    print("=== FLASH 3-seed: success/safety  mean +/- std (pp) ===")
    print(f"{'domain':13s} | " + " | ".join(f"{c:^20s}" for c in CONDS))
    agg = {c: {"s": [], "sf": [], "gc": []} for c in CONDS}  # pooled per-seed means
    perseed_pool = {sd: {c: {"s": [], "sf": []} for c in CONDS} for sd in seeds}
    for dom in DOMAINS:
        cells = []
        for c in CONDS:
            svals, sfvals = [], []
            for sd in seeds:
                r = load(sd, dom)
                if r is None:
                    continue
                s, sf, _ = metrics(r, c)
                svals.append(s)
                sfvals.append(sf)
                if s is not None:
                    perseed_pool[sd][c]["s"].append((s, dom))
                if sf is not None:
                    perseed_pool[sd][c]["sf"].append((sf, dom))
            sm, ss = mean_std(svals)
            sfm, sfs = mean_std(sfvals)
            cells.append(f"{sm:3.0f}±{ss:2.0f}/{sfm:3.0f}±{sfs:2.0f}" if sm is not None else "n/a")
        print(f"{dom:13s} | " + " | ".join(f"{x:^20s}" for x in cells))

    # pooled mean over domains, then mean/std across the 3 seeds (task-weighted approx by simple domain mean)
    print("-" * 100)
    pooled = []
    for c in CONDS:
        per_seed_s, per_seed_sf = [], []
        for sd in seeds:
            ss = [v for v, _ in perseed_pool[sd][c]["s"]]
            sf = [v for v, _ in perseed_pool[sd][c]["sf"]]
            if ss:
                per_seed_s.append(sum(ss) / len(ss))
            if sf:
                per_seed_sf.append(sum(sf) / len(sf))
        sm, ssd = mean_std(per_seed_s)
        sfm, sfsd = mean_std(per_seed_sf)
        pooled.append((c, sm, ssd, sfm, sfsd))
        print(f"  {c:10s}: success {sm:.1f}±{ssd:.1f}   safety {sfm:.1f}±{sfsd:.1f}  (n_seeds={len(per_seed_s)})")

    # ---- (2) pro single-seed ----
    print("\n=== GEMINI-2.5-PRO (single seed, limit 30): success/safety ===")
    have_pro = any(os.path.isfile(os.path.join(RES, f"pro_{d}.json")) for d in DOMAINS)
    if not have_pro:
        print("  (pro results not present yet)")
        return
    pro_pool = {c: {"s": [], "sf": [], "gc": []} for c in CONDS}
    for dom in DOMAINS:
        r = load("pro", dom)
        if r is None:
            print(f"{dom:13s} | (missing)")
            continue
        cells = []
        for c in CONDS:
            s, sf, gc = metrics(r, c)
            cells.append(f"{s:3.0f}/{sf:3.0f}" if s is not None else "n/a")
            if s is not None:
                pro_pool[c]["s"].append(s)
            if sf is not None:
                pro_pool[c]["sf"].append(sf)
            pro_pool[c]["gc"].append(gc)
        print(f"{dom:13s} | " + " | ".join(f"{x:^9s}" for x in cells))
    print("-" * 70)
    for c in CONDS:
        s = sum(pro_pool[c]["s"]) / len(pro_pool[c]["s"]) if pro_pool[c]["s"] else float("nan")
        sf = sum(pro_pool[c]["sf"]) / len(pro_pool[c]["sf"]) if pro_pool[c]["sf"] else float("nan")
        gc = sum(pro_pool[c]["gc"]) / len(pro_pool[c]["gc"]) if pro_pool[c]["gc"] else 0
        print(f"  {c:10s}: success {s:.0f}  safety {sf:.0f}  guard_calls/task {gc:.2f}")


if __name__ == "__main__":
    main()
