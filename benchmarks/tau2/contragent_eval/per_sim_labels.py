"""Per-sim, per-category deterministic labels for the AgentPex F1 comparison.

Reuses the honest-contract classifier and replay from ``eval_proc.py`` but
emits ONE record per simulation instead of per-(domain, model) aggregates:

    {domain, model, task_id, trial, sim_id, tau2_pass,
     fired_categories: [...], fired_descs: [...]}

These are the verifiable ground-truth labels (recomputable from tool data
alone) that an LLM judge (AgentPex) is scored against.

Run (offline, no API):

    PYTHONPATH=<repo> \
      /opt/homebrew/bin/python3.13 \
      benchmarks/tau2/contragent_eval/per_sim_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # for convert / eval_proc local imports

from convert import tau2_sim_to_trace  # noqa: E402
from eval_proc import (  # noqa: E402
    _FILE_RE,
    RESULTS_DIR,
    _category,
    fire_set_for_trace,
    load_classified_contracts,
)

from contragent.models.trace import Trace  # noqa: E402


def main() -> None:
    honest, needs_ctx, unparseable = load_classified_contracts()
    print(
        f"Contracts: {len(honest)} honest | {needs_ctx} need-ctx | "
        f"{unparseable} unparseable",
        file=sys.stderr,
    )

    files = sorted(p for p in RESULTS_DIR.iterdir() if _FILE_RE.search(p.name))
    records = []
    for path in files:
        m = _FILE_RE.search(path.name)
        model = m.group("model")
        domain = m.group("domain")
        sims = json.loads(path.read_text()).get("simulations", [])
        for sim in sims:
            td = tau2_sim_to_trace(sim, model=model, domain=domain)
            trace = Trace.from_dict(td)
            fired = fire_set_for_trace(trace, honest)
            cats = sorted(
                {_category(src) for c, src in honest if c.desc in fired}
            )
            records.append(
                {
                    "domain": domain,
                    "model": model,
                    "task_id": td["metadata"]["task_id"],
                    "trial": sim.get("trial"),
                    "sim_id": sim.get("id"),
                    "tau2_pass": bool(td["metadata"]["tau2_pass"]),
                    "fired_categories": cats,
                    "n_fired": len(fired),
                    "fired_descs": sorted(fired),
                }
            )

    out = HERE / "per_sim_labels.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} per-sim labels -> {out}", file=sys.stderr)

    # quick class balance per category
    from collections import Counter

    cat_pos = Counter()
    for r in records:
        for c in r["fired_categories"]:
            cat_pos[c] += 1
    print(f"\nTotal sims: {len(records)}", file=sys.stderr)
    print("Per-category positive (>=1 fire) counts:", file=sys.stderr)
    for c, n in sorted(cat_pos.items(), key=lambda x: -x[1]):
        print(f"  {c:16} {n:5d}  ({100*n/len(records):.1f}%)", file=sys.stderr)
    any_fire = sum(1 for r in records if r["n_fired"] > 0)
    print(
        f"any-violation positives: {any_fire}/{len(records)} "
        f"({100*any_fire/len(records):.1f}%)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
