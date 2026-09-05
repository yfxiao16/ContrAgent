# AgentDojo: prompt-injection defense (online)

## Claim

Data-flow contracts over the tool-call trace block indirect prompt
injections before the injected action executes, at a per-call cost three
orders of magnitude below model-based defenses, and without a model on the
hot path.

## Setup

AgentDojo records agents solving user tasks in four suites (banking,
workspace, travel, Slack) while attacker text placed in tool results tries
to redirect them. The attack success rate (ASR) is the fraction of attack
traces in which the attacker's goal is reached. ContrAgent replays each
recorded trace and gates every tool call; an attack counts as defended when
a call on the attacker's path is blocked. Utility is measured on
injection-free traces as the fraction of completed tasks that remain
unblocked.

Two libraries are compared:

* **generic**: contracts written from the suite's tool semantics alone
  (no outbound message, transfer, or share to a recipient that does not
  appear in the user's own data; no credential change on an injected turn);
* **trace-learned**: the generic library plus the recipient and URL
  allowlists a deployment maintains independently of any user message,
  read from each suite's environment definition.

## Contracts

`contragent/contracts/benchmark/agentdojo.yaml` (31 contracts over
`called`, `arg_field_has`, `flow`, and `contains`).

## Results (paper Table `tab:agentdojo`, `gpt-4o`)

| Defense | Type | ASR | Overhead per call |
|---|---|---|---|
| No defense | raw agent | 47.7% | 0 |
| spotlighting | prompt-side | 41.7% | extra prompt |
| repeat_user_prompt | prompt-side | 27.8% | extra prompt |
| pi_detector | DeBERTa classifier | 7.95% | ~50 ms |
| tool_filter | LLM prune | 6.84% | ~500 ms |
| **ContrAgent (generic)** | data-flow contracts | **4.93%** | **0.16 ms** |
| LlamaFirewall | ML guardrail | 1.75% | ~100 ms |
| **ContrAgent (trace-learned)** | data-flow contracts | **0.79%** | **0.16 ms** |

## Reproduce

Place AgentDojo's recorded runs under `benchmarks/AgentDojo/runs/<model>/`
(the layout produced by the AgentDojo benchmark runner). The table scores
the `important_instructions` attack class only (629 attack traces on
`gpt-4o-2024-05-13`), the class on which AgentDojo's own defenses were
published, so that all rows share one denominator; `CG_ATTACKS` selects the
classes and defaults to every recorded class.

```bash
cd benchmarks/AgentDojo
# generic row (4.93%): data-flow gate, taint rule
CG_ATTACKS=important_instructions CG_MODE=taint    PYTHONPATH=../.. python contragent_eval/run_eval_dataflow.py gpt-4o-2024-05-13
# trace-learned row (0.79%): taint rule plus the full library on injected content
CG_ATTACKS=important_instructions CG_MODE=taintlib PYTHONPATH=../.. python contragent_eval/run_eval_dataflow.py gpt-4o-2024-05-13
# the raw library without the data-flow gate, all attack classes
PYTHONPATH=../.. python contragent_eval/run_eval.py gpt-4o-2024-05-13
```

`CG_MODE` selects the provenance rule of the data-flow gate in
`run_eval_dataflow.py` (`strict`, `notuser`, `taint`, `taintlib`; the module
docstring defines them). The no-defense row is the `baseASR` column of either
script (47.7%); the two ContrAgent rows are the `DF_ASR` column under the two
modes above (31/629 and 5/629). The default `strict` mode is a different,
stricter gate and is not a reported configuration.
