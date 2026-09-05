# tau²-bench: procedural violation and outcome scoring (offline)

## Claim

Outcome-based scoring hides procedural violations. With a contract library
built from the benchmark's policy documents, ContrAgent measures procedural
compliance deterministically over every recorded trace and exposes the gap
between passing a task and completing it without violating the procedure.

## Setup

tau²-bench evaluates customer-service agents in three domains (retail,
airline, telecom). Its native metric is `pass^k`: the task reaches the
correct final state in all k retries. AgentPex showed that 83% of
reward-1.0 Claude traces still violate a procedural rule. We replay the
published trace matrix (4,464 traces, 12 model x domain cells) against a
library compiled from each domain's `policy.md` and report `joint^k`: the
outcome is correct and no contract fires, in all k retries.

## Contracts

`contragent/contracts/benchmark/tau2_bench.yaml`: 120 contracts (retail 45,
airline 32, telecom 40, cross-domain 3) covering identity gates, ordering,
argument grounding, forbidden transitions, post-action verification, and
same-turn protocol. Four contracts that compare two argument fields
(item-count match, distinct payment method, passenger cap) are excluded
from the reported rates by the harness, as in the paper.

## Results (paper Figure `fig:tau2tax`)

Across the 4,464-trace matrix Claude 3.7 reaches an outcome `pass^4` of
25% to 60% yet a `joint^4` of 0% in all three domains: no task is completed
reliably without at least one procedural violation. The contract library
decides this deterministically for every trace, at 0.83 ms p50 per call
(appendix latency table), where an LLM judge approximates it.

## Reproduce

Place the tau²-bench trace matrix under
`benchmarks/tau2/data/tau2/results/final/` (the layout of the published
runs), then

```bash
PYTHONPATH=. python benchmarks/tau2/contragent_eval/eval_proc.py
```

`eval_proc.py` writes `proc_eval_results.json` with per-cell fire rates and
the `pass^k` / `proc-clean^k` / `joint^k` matrix.
