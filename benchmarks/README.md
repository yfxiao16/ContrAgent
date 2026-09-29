# Benchmark harnesses

Each directory holds the harness for one benchmark of the ContrAgent paper under
`<benchmark>/contragent_eval/`: a converter from the benchmark's native
format to the ContrAgent trace format, the scoring scripts, and (for
SOPBench) the live four-condition enforcement experiment.

Third-party datasets are **not redistributed**. Obtain each from its own
release and place it where the converter expects it, as described below.
The contract libraries are ours and ship with this repository, so a reader
who downloads a benchmark can reproduce the experiment against the same
contracts we used.

## Obtaining the data

| Benchmark | Upstream release | Place under |
|---|---|---|
| SOPBench | <https://github.com/Leezekun/SOPBench> | `SOPBench/` |
| AgentDojo | <https://github.com/ethz-spylab/agentdojo> | `AgentDojo/` |
| R-Judge | <https://github.com/Lordog/R-Judge> | `R-Judge/` |
| tau²-bench | <https://github.com/sierra-research/tau2-bench> | `tau2/` |

Clone into the matching directory, keeping the upstream layout:

```bash
git clone https://github.com/Leezekun/SOPBench.git       benchmarks/SOPBench/upstream
git clone https://github.com/ethz-spylab/agentdojo.git   benchmarks/AgentDojo/upstream
git clone https://github.com/Lordog/R-Judge.git          benchmarks/R-Judge/upstream
git clone https://github.com/sierra-research/tau2-bench.git benchmarks/tau2/upstream
```

Each converter's module docstring states the exact paths it reads; see also
the per-benchmark records under [`../experiments/`](../experiments/).

## Contract libraries

Contracts are versioned in this repository and are never tuned on labels:

| Benchmark | Library |
|---|---|
| SOPBench | [`contragent/contracts/sopbench/`](../contragent/contracts/sopbench/) — one `<domain>.yaml` per domain, plus [`SOPBench/contragent_eval/compile_tree.py`](SOPBench/contragent_eval/compile_tree.py) which compiles the public procedure tree into (A, G) pairs |
| AgentDojo | [`contragent/contracts/benchmark/agentdojo.yaml`](../contragent/contracts/benchmark/agentdojo.yaml) |
| R-Judge | [`contragent/contracts/benchmark/rjudge.yaml`](../contragent/contracts/benchmark/rjudge.yaml) |
| tau²-bench | [`contragent/contracts/benchmark/tau2_bench.yaml`](../contragent/contracts/benchmark/tau2_bench.yaml) |

## Reproducing each experiment

### SOPBench

Offline replay of recorded trajectories against the libraries:

```bash
bash SOPBench/contragent_eval/run.sh bank dmv healthcare hotel library university online_market
```

Online four-condition enforcement (needs a model API key in a repo-root
`.env`). `--limit 40` takes a balanced slice, `pos[:20] + neg[:20]`:

```bash
cd SOPBench && PYTHONPATH=contragent_eval:.:../.. python contragent_eval/live/run.py \
    --domain bank --condition base prompt enforce llm_guard \
    --model gemini-2.5-flash --limit 40 --max-steps 12 --workers 6 \
    --out contragent_eval/live/results/bank.json
python contragent_eval/live/analyze.py contragent_eval/live/results/*.json
```

[`live/sweep_final.sh`](SOPBench/contragent_eval/live/sweep_final.sh) runs
that command over all seven domains, and
[`live/aggregate_seeds.py`](SOPBench/contragent_eval/live/aggregate_seeds.py)
pools repeated sweeps. There is no seed flag: repeated trials differ only
through the hosted model's own nondeterminism, which is why the enforced
condition is stable across them and the model-driven ones are not.

The domains are not all the same size. `university` has only 6 permitted
tasks, so a `--limit 40` slice yields 26 tasks there rather than 40, and
`--limit 0` runs its full pool of 42 (6 permitted, 36 forbidden). The
`university` cell the paper reports is from the full pool; the other domains
are from the balanced 40-task slice.

### AgentDojo

```bash
python AgentDojo/contragent_eval/run_eval.py
python AgentDojo/contragent_eval/run_eval_dataflow.py
```

### R-Judge

```bash
bash R-Judge/contragent_eval/run.sh
python R-Judge/contragent_eval/score_by_category.py
```

### tau²-bench

```bash
python tau2/contragent_eval/eval_proc.py
```

### Appendix tables

```bash
python latency_bench.py            # hot-path latency
python temporal_expressiveness.py  # needs no data
python SOPBench/contragent_eval/appendix_profiles.py  # per-model violation profiles (needs SOPBench traces)
```

## What to expect

The offline benchmarks (R-Judge, tau²-bench, and the AgentDojo and SOPBench
replays) are deterministic: they read fixed upstream traces, no model is
called on the checking path, and a rerun reproduces the reported verdicts
exactly.

The SOPBench live experiment calls a model, so the model-driven conditions
(`base`, `prompt`, `llm_guard`) move by roughly 2 to 4 points between
repeated trials, and the enforced condition by under 1 point, since its
verdict does not depend on the agent. Aggregates are stable. Across our
repeated trials the enforced mean stays at 89 to 90% success and 96 to 98%
safety, and the unguarded base at 91% success and 32% safety. Individual
domain cells can still differ by several points from the table in the
paper, and by more if the hosted model snapshot has changed since, so
compare the aggregates and the ordering between conditions rather than
individual cells.

Converted traces and run outputs are written next to the harness and are
ignored by git.
