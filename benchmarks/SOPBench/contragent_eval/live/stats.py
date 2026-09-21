"""
Aggregate the multi-seed live runs into a paper-ready mean ± std table.

Reads contragent_eval/live/results/seeds/<domain>_s<seed>.json (each a full
3-condition run) and reports, per (domain, condition), the mean and sample
standard deviation of success rate and safety rate across seeds.

Usage: PYTHONPATH=contragent_eval:.:../.. python live/stats.py
"""

import glob
import json
import os
import re
import statistics as st
import sys as _sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = os.path.join(HERE, "results", _sys.argv[1] if len(_sys.argv) > 1 else "seeds")
DOMAINS = ["bank", "dmv", "healthcare", "hotel", "library", "university", "online_market"]
CONDS = ["base", "prompt", "enforce"]


def rates(rows):
    pos = [r for r in rows if r["should_succeed"]]
    neg = [r for r in rows if not r["should_succeed"]]
    su = 100 * sum(1 for r in pos if r["goal_completed"]) / len(pos) if pos else None
    sa = 100 * sum(1 for r in neg if not r["goal_completed"]) / len(neg) if neg else None
    return su, sa


def collect():
    # domain -> cond -> {'su': [..per seed..], 'sa': [...]}
    data = defaultdict(lambda: defaultdict(lambda: {"su": [], "sa": []}))
    for path in sorted(glob.glob(os.path.join(SEEDS, "*_s*.json"))):
        m = re.match(r"(.+)_s(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        dom = m.group(1)
        rows = [r for r in json.load(open(path))["results"] if "error" not in r]
        for c in CONDS:
            su, sa = rates([r for r in rows if r["condition"] == c])
            if su is not None:
                data[dom][c]["su"].append(su)
            if sa is not None:
                data[dom][c]["sa"].append(sa)
    return data


def ms(xs):
    if not xs:
        return None
    m = st.mean(xs)
    s = st.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def fmt(v):
    return f"{v[0]:.0f}±{v[1]:.0f}" if v else "  -  "


def main():
    data = collect()
    nseeds = max((len(data[d][c]["su"]) for d in data for c in CONDS), default=0)
    print(
        f"# Live 3-condition results, mean ± std over {nseeds} seeds "
        f"(gemini-2.5-flash-lite). success / safety (%)\n"
    )
    print(f"{'domain':14s} | {'base':>17s} | {'prompt':>17s} | {'enforce':>17s}")
    print("-" * 76)
    agg = {c: {"su": [], "sa": []} for c in CONDS}
    for d in DOMAINS:
        if d not in data:
            continue
        cells = []
        for c in CONDS:
            su, sa = ms(data[d][c]["su"]), ms(data[d][c]["sa"])
            cells.append(f"{fmt(su)} / {fmt(sa)}")
            if su:
                agg[c]["su"].append(su[0])
            if sa:
                agg[c]["sa"].append(sa[0])
        print(f"{d:14s} | {cells[0]:>17s} | {cells[1]:>17s} | {cells[2]:>17s}")
    print("-" * 76)
    cells = []
    for c in CONDS:
        su = ms(agg[c]["su"])
        sa = ms(agg[c]["sa"])
        cells.append(f"{fmt(su)} / {fmt(sa)}")
    print(f"{'MEAN':14s} | {cells[0]:>17s} | {cells[1]:>17s} | {cells[2]:>17s}")
    print(
        "\n(per-domain cell = mean±std over seeds; MEAN row = mean±std across the "
        "7 per-domain means)"
    )


if __name__ == "__main__":
    main()
