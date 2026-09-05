# Benchmark harnesses

Each directory holds the harness for one benchmark of the paper under
`<benchmark>/contragent_eval/`: a converter from the benchmark's native
format to the ContrAgent trace format, the scoring scripts, and (for
SOPBench) the live four-condition enforcement experiment. Third-party
datasets are not redistributed; obtain them from the benchmark's own
release and place them where each converter expects them (see the module
docstrings and the records under `../experiments/`).

* `SOPBench/` – offline replay (`run.sh`) and live enforcement (`live/run.py`).
* `AgentDojo/` – replay of recorded AgentDojo runs (`run_eval.py`, `run_eval_dataflow.py`).
* `R-Judge/` – conversion and evaluation of R-Judge records (`run.sh`, `score_by_category.py`).
* `tau2/` – procedural-violation scoring of the tau²-bench trace matrix (`eval_proc.py`).
* `latency_bench.py` – hot-path latency table (appendix).
* `temporal_expressiveness.py` – temporal-expressiveness table (appendix); needs no data.

Converted traces and run outputs are written next to the harness and are
ignored by git.
