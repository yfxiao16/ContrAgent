"""Offline procedure-violation eval: replay tau2 traces through ContrAgent.

Loads the bundled ``contragent/contracts/benchmark/tau2_bench.yaml``
contract library, converts every tau2 simulation transcript to a native
ContrAgent trace (see ``convert.py``), and replays each trace through the
contracts with ``TraceVerifier``. Reports per-(domain, model) procedure-
violation fire-rates -- the "proc-clean vs pass" idea from
``experiments/tau2.md``.

Run (offline, no API/LLM):

    PYTHONPATH=<repo> \
      python3 \
      benchmarks/tau2/contragent_eval/eval_proc.py

Honest-evaluation policy
------------------------
Contracts split into three classes by their atom dependencies:

  * HONEST -- evaluable from real tool names / args / ordering alone.
    Includes all contracts over tool names, argument values and counts
    (precedence, count limits, argument allow/deny lists, value ranges) plus pure-LTL ones
    over ``called`` / ``X(called)`` (ordering, post-action verification)
    and the structural ``same_turn_text_and_tool_call`` protocol flag.

  * NEEDS_CTX -- depend on policy-derived ctx facts (user_authenticated,
    target_order_status, owner_match, account_status, ...). These cannot
    be derived from the transcript without re-implementing the domain
    policy against the environment DB -- the semantic tagging the
    converter refuses to do. SKIPPED and counted separately.

  * UNPARSEABLE -- the 4 Cat-C ``ArgLength``/``ArgValue`` contracts whose
    infix form the public ``parse_repr`` does not yet accept. SKIPPED.

Only HONEST contracts contribute to the reported fire-rates.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from convert import tau2_sim_to_trace  # noqa: E402

from contragent.config import ConstraintEntry, _compile_single  # noqa: E402
from contragent.formulas.formula import Atom, Var  # noqa: E402
from contragent.models.agent import Agent  # noqa: E402
from contragent.models.contract import Contract  # noqa: E402
from contragent.models.trace import Trace  # noqa: E402
from contragent.runtime.verifier import TraceVerifier, _raw_formula  # noqa: E402

LIBRARY = REPO_ROOT / "contragent" / "contracts" / "benchmark" / "tau2_bench.yaml"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "tau2" / "results" / "final"

# Canonical leaderboard cells: 4 models x 3 domains, default/base variant
# (telecom-workflow + ablation variants excluded).
_FILE_RE = re.compile(
    r"(?P<model>.+?)_(?P<domain>retail|airline|telecom)_(?P<variant>default|base)_"
    r"gpt-4\.1-2025-04-14_4trials\.json$"
)


# --------------------------------------------------------------------------
# Contract loading (tolerant: compile each entry independently, classify)
# --------------------------------------------------------------------------


def _atoms(node, acc: set[str]) -> None:
    if node is None:
        return
    if hasattr(node, "formula"):
        _atoms(node.formula, acc)
        return
    if isinstance(node, Atom):
        acc.add(node.predicate)
        return
    if isinstance(node, Var):
        acc.add(node.name)
        return
    for a in ("child", "left", "right"):
        c = getattr(node, a, None)
        if c is not None:
            _atoms(c, acc)


CAT_C_EXCLUDED = {
    "modify_pending_order_items: item_ids and new_item_ids must have the same length",
    "exchange_delivered_order_items: item_ids and new_item_ids must have the same length",
    "modify_pending_order_payment.payment_method_id must differ from the order's original method",
    "book_reservation must have at most 5 passengers",
}


def load_classified_contracts():
    """Return (honest, needs_ctx_count, unparseable_count).

    ``honest`` is a list of (Contract, source_category) tuples for the
    contracts that are evaluable from real tool data alone.
    """
    raw = yaml.safe_load(LIBRARY.read_text())
    entries = raw["agents"]["*"]["contracts"]
    agent = Agent(id="*")

    honest: list[tuple[Contract, str]] = []
    needs_ctx = 0
    unparseable = 0

    for it in entries:
        g = it["G"]
        desc = it.get("desc", "")
        source = g.get("source", "")
        if desc in CAT_C_EXCLUDED:
            # The four Cat-C contracts compare two argument fields (item-count
            # match, distinct payment method, passenger cap). The paper's
            # reported fire rates exclude them, so they are kept out here too.
            unparseable += 1
            continue
        ce = ConstraintEntry(ltl=g.get("ltl"))
        try:
            formula = _compile_single(ce, None, None)
        except Exception:
            unparseable += 1
            continue

        preds: set[str] = set()
        _atoms(_raw_formula(formula), preds)
        # ctx / ctx_value / ctx_matches => policy-derived fact, EXCEPT the
        # same_turn_text_and_tool_call flag which is a purely structural
        # property of one assistant message (text + tool_call) that the
        # converter emits honestly. Detect it by inspecting the raw ltl.
        ltl_txt = g.get("ltl") or ""
        structural_ctx_only = "same_turn_text_and_tool_call" in ltl_txt
        if any(p.startswith("ctx") for p in preds) and not structural_ctx_only:
            needs_ctx += 1
            continue

        contract = Contract(agent=agent, guarantee=formula, desc=desc)
        honest.append((contract, source))

    return honest, needs_ctx, unparseable


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


def fire_set_for_trace(trace: Trace, contracts) -> set[str]:
    """Return the set of contract descs that fire (are violated) on a trace."""
    v = TraceVerifier()
    v.sync_from_contracts(trace, [c for c, _ in contracts])
    fired: set[str] = set()
    for contract, _src in contracts:
        verdict = v.check_contract(contract)
        if not verdict.holds:
            fired.add(contract.desc)
    return fired


def _category(source: str) -> str:
    # source: library:benchmark.tau2/<domain>/<category>
    return source.rsplit("/", 1)[-1] if "/" in source else source


def evaluate():
    honest, needs_ctx, unparseable = load_classified_contracts()
    print(
        f"Contracts: {len(honest)} honest (evaluated)  |  "
        f"{needs_ctx} need derived ctx (skipped)  |  "
        f"{unparseable} unparseable Cat-C (skipped)"
    )
    print(f"Total in library: {len(honest) + needs_ctx + unparseable}\n")

    files = sorted(p for p in RESULTS_DIR.iterdir() if _FILE_RE.search(p.name))
    rows = []
    # category fire counts per cell
    cat_rows = []

    for path in files:
        m = _FILE_RE.search(path.name)
        model = m.group("model")
        domain = m.group("domain")
        results = json.loads(path.read_text())
        sims = results.get("simulations", [])

        n = len(sims)
        n_pass = 0
        n_fire = 0  # sims that fire >=1 honest contract
        n_pass_and_fire = 0  # blind-spot: tau2-pass but procedure-violating
        n_fail_and_fire = 0  # proc-recall: tau2-fail caught
        n_fail = 0
        cat_hits: dict[str, int] = defaultdict(int)
        # group sims by task for proc-clean^k
        by_task: dict[int, list[bool]] = defaultdict(list)  # task -> [clean?]
        by_task_pass: dict[int, list[bool]] = defaultdict(list)

        for sim in sims:
            td = tau2_sim_to_trace(sim, model=model, domain=domain)
            trace = Trace.from_dict(td)
            fired = fire_set_for_trace(trace, honest)
            is_pass = bool(td["metadata"]["tau2_pass"])
            clean = len(fired) == 0
            if is_pass:
                n_pass += 1
            else:
                n_fail += 1
            if fired:
                n_fire += 1
                if is_pass:
                    n_pass_and_fire += 1
                else:
                    n_fail_and_fire += 1
            # category fingerprint
            cats_this = set()
            for c, src in honest:
                if c.desc in fired:
                    cats_this.add(_category(src))
            for cat in cats_this:
                cat_hits[cat] += 1
            task = td["metadata"]["task_id"]
            by_task[task].append(clean)
            by_task_pass[task].append(is_pass)

        # proc-clean^k: fraction of tasks where ALL trials were clean
        tasks = list(by_task)
        proc_clean_k = sum(1 for t in tasks if all(by_task[t])) / len(tasks) if tasks else 0.0
        pass_k = sum(1 for t in tasks if all(by_task_pass[t])) / len(tasks) if tasks else 0.0
        joint_k = (
            sum(1 for t in tasks if all(by_task[t]) and all(by_task_pass[t])) / len(tasks)
            if tasks
            else 0.0
        )

        rows.append(
            {
                "domain": domain,
                "model": model,
                "sims": n,
                "tau2_pass_sims": n_pass,
                "fire_rate": n_fire / n if n else 0.0,
                "blindspot": (n_pass_and_fire / n_pass) if n_pass else 0.0,
                "proc_recall": (n_fail_and_fire / n_fail) if n_fail else 0.0,
                "pass_k": pass_k,
                "proc_clean_k": proc_clean_k,
                "joint_k": joint_k,
            }
        )
        cat_rows.append((domain, model, n, dict(cat_hits)))

    return rows, cat_rows


def main():
    rows, cat_rows = evaluate()

    order = {"retail": 0, "airline": 1, "telecom": 2}
    rows.sort(key=lambda r: (order.get(r["domain"], 9), r["model"]))

    print(
        f"{'domain':8} {'model':28} {'sims':>5} {'fire%':>6} "
        f"{'blind%':>7} {'recall%':>8} {'pass^4':>7} {'clean^4':>8} {'joint^4':>8}"
    )
    print("-" * 96)
    for r in rows:
        print(
            f"{r['domain']:8} {r['model']:28} {r['sims']:5d} "
            f"{100 * r['fire_rate']:6.1f} {100 * r['blindspot']:7.1f} "
            f"{100 * r['proc_recall']:8.1f} {100 * r['pass_k']:7.1f} "
            f"{100 * r['proc_clean_k']:8.1f} {100 * r['joint_k']:8.1f}"
        )

    print("\nPer-category sim-level fire rates (fraction of sims firing >=1 in cat):")
    cat_rows.sort(key=lambda x: (order.get(x[0], 9), x[1]))
    all_cats = sorted({c for _, _, _, ch in cat_rows for c in ch})
    hdr = f"{'domain':8} {'model':28} " + " ".join(f"{c[:10]:>11}" for c in all_cats)
    print(hdr)
    print("-" * len(hdr))
    for domain, model, n, ch in cat_rows:
        cells = " ".join(f"{100 * ch.get(c, 0) / n:10.1f}%" for c in all_cats)
        print(f"{domain:8} {model:28} {cells}")

    out = Path(__file__).resolve().parent / "proc_eval_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
