# Result files

The machine-readable summaries behind the numbers in the experiment
records. Each file is the output of the harness named next to it; none
contains benchmark task text, only counts and rates.

| File | Produced by | Backs |
|---|---|---|
| `rjudge_eval.json` | `contragent eval benchmarks/R-Judge/contragent_eval/traces --config contragent/contracts/benchmark/rjudge.yaml --agent "*" --json` (regenerated 2026-09-29 from the converted traces) | [r-judge.md](../r-judge.md): 571 records, precision 97.0%, recall 87.0% (`overall`), and per-contract counts |
| `rjudge_by_category.txt` | `benchmarks/R-Judge/contragent_eval/score_by_category.py` (same traces, grounding without content atoms) | [r-judge.md](../r-judge.md): the per-category breakdown and the decidable-versus-semantic split of the unsafe set; its overall recall differs from `rjudge_eval.json` because it grounds without content atoms |
| `tau2_proc_eval.json` | `benchmarks/tau2/contragent_eval/eval_proc.py` (2026-09-20) | [tau2.md](../tau2.md): per domain and model, simulations, `pass^k`, `proc_clean^k`, `joint^k`, fire rate, blind spot |
| `agentdojo_raw_library.json` | `benchmarks/AgentDojo/contragent_eval/run_eval.py` (2026-09-20): the raw library without the data-flow gate, 28 recorded model runs | [agentdojo.md](../agentdojo.md): per model, base attack success rate, attack success rate under ContrAgent, reduction, utility false positives |

The SOPBench live experiment writes per-task files under
`benchmarks/SOPBench/contragent_eval/live/results/` (260 files, 21 MB,
June 2026: `final_<domain>.json` for the first seed of every condition,
`s2_`/`s3_` repeats, and the `<domain>_enffix_f_s{1,2,3}.json` enforce
reruns after the enforcement fixes, whose three-seed mean is the 89–90 /
96–98 reported in [`benchmarks/README.md`](../../benchmarks/README.md)
under "What to expect"). They contain the benchmark's task text and are
kept outside the repository. To regenerate them, place the release as
[`benchmarks/README.md`](../../benchmarks/README.md) describes, run
`live/sweep_final.sh` (a model key in `.env`; one sweep is seven domains
by four conditions), and pool repeated sweeps with `aggregate_seeds.py`.
