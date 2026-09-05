"""Build a stratified sample of tau2 sims for the AgentPex F1 comparison.

For each (domain, model) cell, draw N sims (seeded). Group by domain and
write one tau-sq-format file per domain (shared policy/tools -> one spec
extraction), plus a manifest mapping sim_id -> deterministic GT labels.

Output dir: this script's directory.
  retail_sample.json / airline_sample.json / telecom_sample.json  (tau-sq)
  manifest.json   { sim_id: {domain, model, gt_categories, gt_any} }

Run:
  /opt/homebrew/bin/python3.13 build_sample.py [N_PER_CELL]
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent  # contragent_eval
RESULTS = EVAL_DIR.parent / "data" / "tau2" / "results" / "final"
LABELS = EVAL_DIR / "per_sim_labels.json"

N_PER_CELL = int(sys.argv[1]) if len(sys.argv) > 1 else 25
SEED = 0

# canonical cell file names -> regex pieces (default/base variant only)
import re

_FILE_RE = re.compile(
    r"(?P<model>.+?)_(?P<domain>retail|airline|telecom)_(?P<variant>default|base)_"
    r"gpt-4\.1-2025-04-14_4trials\.json$"
)


def main() -> None:
    labels = json.loads(LABELS.read_text())
    by_id = {r["sim_id"]: r for r in labels}

    # cell -> list of sim_ids
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in labels:
        cells[(r["domain"], r["model"])].append(r["sim_id"])

    rng = random.Random(SEED)
    sampled_ids: set[str] = set()
    for cell, ids in sorted(cells.items()):
        ids_sorted = sorted(ids)  # deterministic base order
        pick = rng.sample(ids_sorted, min(N_PER_CELL, len(ids_sorted)))
        sampled_ids.update(pick)
    print(f"Sampled {len(sampled_ids)} sims "
          f"({N_PER_CELL}/cell x {len(cells)} cells)", file=sys.stderr)

    # load raw files, index sims by id, and keep one info/tasks per domain
    domain_info: dict[str, dict] = {}
    domain_tasks: dict[str, dict] = {}
    domain_sims: dict[str, list] = defaultdict(list)

    for path in sorted(RESULTS.iterdir()):
        m = _FILE_RE.search(path.name)
        if not m:
            continue
        domain = m.group("domain")
        raw = json.loads(path.read_text())
        if domain not in domain_info:
            domain_info[domain] = raw["info"]
            domain_tasks[domain] = {}
        for t in raw.get("tasks", []):
            tid = t.get("id") if isinstance(t, dict) else None
            if tid is not None:
                domain_tasks[domain][tid] = t
        for sim in raw["simulations"]:
            if sim.get("id") in sampled_ids:
                domain_sims[domain].append(sim)

    manifest = {}
    for domain in domain_sims:
        sims = domain_sims[domain]
        out = {
            "info": domain_info[domain],
            "tasks": list(domain_tasks[domain].values()),
            "simulations": sims,
        }
        (HERE / f"{domain}_sample.json").write_text(json.dumps(out))
        for sim in sims:
            r = by_id[sim["id"]]
            manifest[sim["id"]] = {
                "domain": r["domain"],
                "model": r["model"],
                "gt_categories": r["fired_categories"],
                "gt_any": r["n_fired"] > 0,
            }
        print(f"  {domain}: {len(sims)} sims -> {domain}_sample.json",
              file=sys.stderr)

    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pos = sum(1 for v in manifest.values() if v["gt_any"])
    print(f"manifest: {len(manifest)} sims, any-violation pos={pos} "
          f"({100*pos/len(manifest):.1f}%)", file=sys.stderr)
    # per-category positives in the sample
    catc = defaultdict(int)
    for v in manifest.values():
        for c in v["gt_categories"]:
            catc[c] += 1
    for c, n in sorted(catc.items(), key=lambda x: -x[1]):
        print(f"  sample {c:16} pos={n}", file=sys.stderr)


if __name__ == "__main__":
    main()
