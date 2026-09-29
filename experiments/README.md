# Experiment records

One record per benchmark in the ContrAgent paper
([arXiv:2609.18128](https://arxiv.org/abs/2609.18128)). Each record states
the claim the benchmark supports, the contract library used, how the data is
obtained and converted, how the harness is run, and the numbers the paper
reports.

| Benchmark | Role | Record |
|---|---|---|
| SOPBench | enforcement (online) | [sopbench.md](sopbench.md) |
| AgentDojo | enforcement (online) | [agentdojo.md](agentdojo.md) |
| R-Judge | evaluation (offline) | [r-judge.md](r-judge.md) |
| tau²-bench | evaluation (offline) | [tau2.md](tau2.md) |

Shared method:

* Every conversion script is a faithful translator from the benchmark's
  native format to the ContrAgent trace format (real tool names, arguments,
  and contents; no labels are read).
* Every contract is written as an ALTL<sub>f</sub> formula over the
  interaction predicates and lives in a versioned library file under
  `contragent/contracts/`. Libraries are built from the benchmark's public
  policy material, never tuned on labels.
* Verdicts are deterministic: replaying a trace always yields the same
  verdict, per-contract attribution, and first violating event. No model is
  called on the checking path.
* Hot-path latency (appendix table) is measured with
  `benchmarks/latency_bench.py`; the temporal-expressiveness table with
  `benchmarks/temporal_expressiveness.py`.
