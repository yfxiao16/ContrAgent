# SOPBench: standard-operating-procedure enforcement (online)

## Claim

Putting a procedure in the prompt does not make an agent follow it reliably,
and a second model that judges every call over-blocks. A deterministic
supervisor enforces the same procedure at the action boundary without
lowering task success, and its safety verdict does not depend on the agent.

## Setup

SOPBench evaluates whether an agent follows an explicit standard operating
procedure across seven customer-service domains (bank, DMV, healthcare,
hotel, library, university, online market). Each task ships a constraint
graph, numeric thresholds, an initial database, and a label saying whether
the procedure permits the goal. The environment does not self-enforce, so
compliance must come from the agent or from the supervisor.

Four conditions share one base model (`gemini-2.5-flash`, 40 tasks per
domain, three seeded trials):

* **base**: no procedure given;
* **prompt**: the procedure rendered into the system prompt;
* **LLM-guard**: a second model judges every call against the procedure;
* **ContrAgent**: every call is gated by the domain's contract library; a
  blocked call returns the violated contract to the agent, which may retry.

Metrics are **success** (goal completion on permitted tasks) and **safety**
(correct blocking on forbidden tasks), in percent.

## Contracts

`contragent/contracts/sopbench/<domain>.yaml`, one library per domain,
compiled from the public procedure's gate/chain tree by
`contragent/contracts/sopbench/compile_tree.py` (`and`/`chain` to
conjunction, `gate`/`or` to disjunction). Every entry is a formula over
`called`, `arg_value`, and the procedure's numeric thresholds.

## Results (paper Table `tab:sopbench-main`)

| domain | base | prompt | LLM-guard | ContrAgent |
|---|---|---|---|---|
| bank | 70 / 65 | 58 / 100 | 42 / 100 | **70 / 100** |
| DMV | 100 / 60 | 72 / 98 | 20 / 100 | **100 / 100** |
| healthcare | 90 / 48 | 52 / 95 | 20 / 100 | **83 / 100** |
| hotel | 100 / 0 | 30 / 98 | 15 / 100 | **100 / 100** |
| library | 80 / 25 | 48 / 95 | 22 / 98 | **80 / 100** |
| university | 100 / 8 | 92 / 82 | 33 / 88 | **100 / 83** |
| online market | 100 / 20 | 92 / 90 | 15 / 98 | **100 / 100** |
| **mean** | 91 / 32 | 64 / 94 | 24 / 98 | **90 / 98** |
| avg. runtime per task | 1.96 s | +0.905 s | +1.34 s | +0.135 s |

Cells are success / safety. Enforcement holds success at the base level
while raising safety from 32% to 98%; prompting and the LLM-guard reach high
safety only by over-blocking. The two residual gaps trace to the agent: in
healthcare it does not retry after a block, and in university it makes a
permitted change to a target that the outcome-based scorer cannot tell
apart from the forbidden one. Enforcement varies by at most 0.4 pp across
seeds; the model-driven conditions vary by 2 to 4 pp. On a weaker base model
(`gemini-2.5-flash-lite`) prompt safety collapses from 94% to 45% while
enforced safety stays at 98%, because the verdict does not depend on the
agent's model.

## Reproduce

Obtain the SOPBench release (environments, procedures, tasks) and place it
under `benchmarks/SOPBench/` as described in
`benchmarks/SOPBench/contragent_eval/convert.py`.

```bash
# offline: replay recorded trajectories against the libraries
bash benchmarks/SOPBench/contragent_eval/run.sh bank dmv healthcare hotel library university online_market

# online: the four-condition live experiment (needs a model API key in .env)
cd benchmarks/SOPBench && PYTHONPATH=contragent_eval:.:../.. python contragent_eval/live/run.py \
    --domain bank --condition base prompt enforce llm_guard --seed 0
python benchmarks/SOPBench/contragent_eval/live/aggregate_seeds.py
```
