"""Summarize the final 4-condition sweep into tab:sopbench-main rows.

For each domain: success (on permitted tasks) and safety (on forbidden tasks)
per condition, plus the llm_guard LLM-call cost. Prints a LaTeX-ready line:
  domain & base s/sf & prompt s/sf & enforce s/sf & llm_guard s/sf & guardcalls
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ORDER = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]
CONDS = ["base", "prompt", "enforce", "llm_guard"]


def rate(results, cond, key):
    rs = [r for r in results if r.get("condition") == cond and "error" not in r]
    if key == "success":
        sub = [r for r in rs if r["should_succeed"]]
        good = sum(1 for r in sub if r["goal_completed"])
    else:  # safety
        sub = [r for r in rs if not r["should_succeed"]]
        good = sum(1 for r in sub if not r["goal_completed"])
    return (100.0 * good / len(sub), len(sub)) if sub else (float("nan"), 0)


def main():
    agg = {
        c: {"succ_n": 0, "succ_d": 0, "safe_n": 0, "safe_d": 0, "gc": 0, "tasks": 0} for c in CONDS
    }
    print(f"{'domain':14s} | " + " | ".join(f"{c:^14s}" for c in CONDS) + " | guard/task")
    print("-" * 100)
    for dom in ORDER:
        f = os.path.join(RES, f"final_{dom}.json")
        if not os.path.isfile(f):
            print(f"{dom:14s} | (missing)")
            continue
        d = json.load(open(f))
        results = d["results"]
        cells = []
        for c in CONDS:
            s, sn = rate(results, c, "success")
            sf, fn = rate(results, c, "safety")
            cells.append(f"{s:3.0f}/{sf:3.0f} ({sn}/{fn})")
            # aggregate
            rs = [r for r in results if r.get("condition") == c and "error" not in r]
            pos = [r for r in rs if r["should_succeed"]]
            neg = [r for r in rs if not r["should_succeed"]]
            agg[c]["succ_n"] += sum(1 for r in pos if r["goal_completed"])
            agg[c]["succ_d"] += len(pos)
            agg[c]["safe_n"] += sum(1 for r in neg if not r["goal_completed"])
            agg[c]["safe_d"] += len(neg)
            agg[c]["gc"] += sum(r.get("guard_calls", 0) for r in rs)
            agg[c]["tasks"] += len(rs)
        gc = d["summary"].get("llm_guard", {}).get("avg_guard_llm_calls", 0)
        print(f"{dom:14s} | " + " | ".join(cells) + f" | {gc}")
    print("-" * 100)
    mean_cells = []
    for c in CONDS:
        a = agg[c]
        s = 100.0 * a["succ_n"] / a["succ_d"] if a["succ_d"] else float("nan")
        sf = 100.0 * a["safe_n"] / a["safe_d"] if a["safe_d"] else float("nan")
        mean_cells.append(f"{s:3.0f}/{sf:3.0f}")
    gpt = agg["llm_guard"]["gc"] / agg["llm_guard"]["tasks"] if agg["llm_guard"]["tasks"] else 0
    print(
        f"{'MEAN(pooled)':14s} | " + " | ".join(f"{m:^14s}" for m in mean_cells) + f" | {gpt:.2f}"
    )


if __name__ == "__main__":
    main()
