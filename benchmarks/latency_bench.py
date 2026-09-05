"""Hot-path latency of the ContrAgent before-call deterministic check.

Fills the SOPBench / R-Judge / tau2 rows of the paper's latency table. Uses
each workload's OWN contract bundle + converted traces and times the SAME
primitive an online guard runs at a tool call: ground the trace-so-far and
evaluate every loaded det contract.

  * per-call workloads (SOPBench, tau2): at each successive tool call we time
    one full contract-set evaluation on the prefix trace -> one sample/call.
  * per-record workloads (R-Judge): one evaluation on the whole transcript.

No LLM, no network. Run:

    PYTHONPATH=<repo> \
      /opt/homebrew/opt/python@3.13/bin/python3.13 benchmarks/latency_bench.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

warnings.simplefilter("ignore")


def _pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _summ(name, c_lo, c_hi, samples):
    ms = [s * 1000.0 for s in samples]
    print(
        f"{name:24s} C={c_lo}-{c_hi:<3} n={len(ms):6d} "
        f"p50={_pct(ms,0.50):.4f} p95={_pct(ms,0.95):.4f} p99={_pct(ms,0.99):.4f} ms"
    )
    return {
        "workload": name,
        "C_lo": c_lo,
        "C_hi": c_hi,
        "n": len(ms),
        "p50": round(_pct(ms, 0.50), 4),
        "p95": round(_pct(ms, 0.95), 4),
        "p99": round(_pct(ms, 0.99), 4),
    }


# --------------------------------------------------------------------------
# config-loader path (SOPBench, R-Judge) -- same schema as AgentDojo
# --------------------------------------------------------------------------
def load_det_contracts(yaml_path):
    from contragent.eval_runner import resolve_entry
    from contragent.config import load_config

    cfg = load_config(yaml_path)
    agent = cfg.agents.get("*") or next(iter(cfg.agents.values()))
    parsed = []
    for ce in agent.contracts:
        for part in (ce.assumption, ce.guarantee):
            if part is None:
                continue
            for e in part if isinstance(part, list) else [part]:
                _nl, p = resolve_entry(e)
                if p is not None:
                    parsed.append(p)
    return parsed


def _tool_call_idxs(events):
    """Indices of tool_call events (the points an online guard fires)."""
    return [
        i
        for i, e in enumerate(events)
        if (e.get("type") or e.get("event_type")) == "tool_call"
    ]


def _per_call_samples(parsed, trace_dict, samples):
    """Time the REAL online hot path: one persistent verifier per trace, at
    each tool call we incrementally sync the newly-arrived events and step
    every contract's DFA. This is O(ΔN) per call, not a full re-ground."""
    from contragent.runtime.verifier import TraceVerifier
    from contragent.tracer.grounding import collect_content_atoms
    from contragent.models.trace import Trace

    content_atoms = collect_content_atoms([p for p in parsed]) or None
    evs = trace_dict.get("events", [])
    base = {k: v for k, v in trace_dict.items() if k != "events"}
    v = TraceVerifier()
    for idx in _tool_call_idxs(evs):
        t = Trace.from_dict({**base, "events": evs[: idx + 1]})
        t0 = time.perf_counter()
        v.sync(t, content_atoms)  # incremental: grounds only the new events
        for p in parsed:
            try:
                v.check(p)
            except Exception:  # noqa: BLE001
                continue
        samples.append(time.perf_counter() - t0)


def bench_sopbench():
    cdir = os.path.join(REPO, "benchmarks", "SOPBench", "contragent_eval", "contracts")
    tdir = os.path.join(REPO, "benchmarks", "SOPBench", "contragent_eval", "traces")
    samples, c_lo, c_hi = [], 99, 0
    for dom in sorted(os.listdir(tdir)):
        ydir = os.path.join(cdir, f"{dom}.yaml")
        if not os.path.isfile(ydir):
            continue
        parsed = load_det_contracts(ydir)
        c_lo, c_hi = min(c_lo, len(parsed)), max(c_hi, len(parsed))
        for f in sorted(glob.glob(os.path.join(tdir, dom, "*.json")))[:30]:
            try:
                d = json.load(open(f))
            except (ValueError, OSError):
                continue
            _per_call_samples(parsed, d, samples)
    return _summ("SOPBench (per call)", c_lo, c_hi, samples)


def bench_rjudge():
    from contragent.runtime.verifier import TraceVerifier
    from contragent.tracer.grounding import collect_content_atoms
    from contragent.models.trace import Trace

    yaml_path = os.path.join(REPO, "benchmarks", "R-Judge", "contragent_eval", "contracts.yaml")
    tdir = os.path.join(REPO, "benchmarks", "R-Judge", "contragent_eval", "traces")
    parsed = load_det_contracts(yaml_path)
    content_atoms = collect_content_atoms([p for p in parsed]) or None
    samples = []
    for f in sorted(glob.glob(os.path.join(tdir, "*.json"))):
        try:
            d = json.load(open(f))
        except (ValueError, OSError):
            continue
        t = Trace.from_dict(d)
        v = TraceVerifier()
        t0 = time.perf_counter()
        v.sync(t, content_atoms)
        for p in parsed:
            try:
                v.check(p)
            except Exception:  # noqa: BLE001
                continue
        samples.append(time.perf_counter() - t0)
    return _summ("R-Judge (per record)", len(parsed), len(parsed), samples)


# --------------------------------------------------------------------------
# tau2 path (TraceVerifier, reuses eval_proc machinery)
# --------------------------------------------------------------------------
def bench_tau2():
    t2dir = os.path.join(REPO, "benchmarks", "tau2", "contragent_eval")
    sys.path.insert(0, t2dir)
    import eval_proc  # noqa: E402
    from contragent.models.trace import Trace  # noqa: E402
    from convert import tau2_sim_to_trace  # noqa: E402

    from contragent.runtime.verifier import TraceVerifier, _collect_det_formulas
    from contragent.tracer.grounding import collect_content_atoms

    honest, _nc, _un = eval_proc.load_classified_contracts()
    contracts = [c for c, _ in honest]
    content_atoms = collect_content_atoms(_collect_det_formulas(contracts)) or None
    c = len(contracts)
    samples = []
    files = sorted(p for p in eval_proc.RESULTS_DIR.iterdir() if eval_proc._FILE_RE.search(p.name))
    for path in files[:6]:  # a few leaderboard cells is plenty for percentiles
        m = eval_proc._FILE_RE.search(path.name)
        model, domain = m.group("model"), m.group("domain")
        results = json.loads(path.read_text())
        for sim in results.get("simulations", [])[:40]:
            td = tau2_sim_to_trace(sim, model=model, domain=domain)
            evs = td.get("events", [])
            base = {k: v for k, v in td.items() if k != "events"}
            v = TraceVerifier()
            for idx in _tool_call_idxs(evs):
                tr = Trace.from_dict({**base, "events": evs[: idx + 1]})
                t0 = time.perf_counter()
                v.sync(tr, content_atoms)
                for contract in contracts:
                    try:
                        v.check_contract(contract)
                    except Exception:  # noqa: BLE001
                        continue
                samples.append(time.perf_counter() - t0)
    return _summ("tau2 (per call)", c, c, samples)


def main():
    rows = []
    print("Warming up...")
    # warm import/JIT caches so cold-start doesn't skew p50
    rows.append(bench_sopbench())
    rows.append(bench_rjudge())
    try:
        rows.append(bench_tau2())
    except Exception as e:  # noqa: BLE001
        print(f"tau2 bench skipped: {type(e).__name__}: {e}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_results.json")
    json.dump(rows, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
