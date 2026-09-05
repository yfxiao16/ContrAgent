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

## Results (paper Table `tab:agentdojo`, `gpt-4o-2024-05-13`, `important_instructions`)

Utility is the fraction of the 124 injection-free tasks completed, computed
from AgentDojo's published runs for every row.

| Defense | Type | ASR | Utility | Overhead per call |
|---|---|---|---|---|
| No defense | raw agent | 47.7% | 72.6% | 0 |
| spotlighting | prompt-side | 41.7% | 75.0% | extra prompt |
| repeat_user_prompt | prompt-side | 27.8% | 85.5% | extra prompt |
| **ContrAgent (no detector)** | data-flow gate, `CG_MODE=strict` | **11.1%** | **71.8%** | **0.16 ms** |
| pi_detector | DeBERTa classifier | 7.95% | 41.9% | ~50 ms |
| tool_filter | LLM prune | 6.84% | 71.8% | ~500 ms |
| pi_detector + ContrAgent (trace-learned) | classifier flags as the untrusted span, `CG_MODE=taintlib` | 2.2% | 55.6% | ~50 ms |
| LlamaFirewall | ML guardrail | 1.75% | n/a | ~100 ms |
| **ContrAgent (trace-learned)** | recorded injection text as the untrusted span, `CG_MODE=taintlib` | **0.79%** | **72.6%** | **0.16 ms** |

Figure 5 (22 models, all attack classes pooled): no detector 18.0% -> 5.3%
ASR at 0.8% utility FP; trace-learned 1.7% at 0.3% FP.

What each ContrAgent row assumes about the untrusted span:

* *no detector*: every tool output is untrusted unless the user named the
  value; nothing is known about the attacks (`strict`).
* *trace-learned*: the untrusted span is the injected text recorded by the
  benchmark, i.e. an ideal detector, and the library includes deny lists
  mined from the attack traces (`taintlib` with the `injections` record).
  The `taint` mode alone (ideal span, no library) gives 4.93% / 72.6% and is
  not reported.
* *pi_detector + ContrAgent*: the untrusted span is the set of tool outputs
  that AgentDojo's own classifier (`protectai/deberta-v3-base-prompt-injection-v2`,
  threshold 0.5) flags when run over the recorded tool outputs
  (`pi_detector_taint.py`, `CG_TAINT`). On gpt-4o the classifier flags at
  least one output in 541 of 629 attack traces and in 78 of 124 injection-free
  traces; the flags arm the contracts instead of deleting data, which is why
  the combination keeps more utility than the classifier alone. With the
  classifier and no library (`taint`) the gate gives 12.7% / 71.8%.

## Reproduce

Place AgentDojo's recorded runs under `benchmarks/AgentDojo/runs/<model>/`
(the layout produced by the AgentDojo benchmark runner). The table scores
the `important_instructions` attack class only (629 attack traces on
`gpt-4o-2024-05-13`), the class on which AgentDojo's own defenses were
published, so that all rows share one denominator; `CG_ATTACKS` selects the
classes and defaults to every recorded class.

```bash
cd benchmarks/AgentDojo
# no-detector row (11.1%)
CG_ATTACKS=important_instructions CG_MODE=strict   PYTHONPATH=../.. python contragent_eval/run_eval_dataflow.py gpt-4o-2024-05-13
# trace-learned row (0.79%): recorded injected text as the untrusted span, plus the full library
CG_ATTACKS=important_instructions CG_MODE=taintlib PYTHONPATH=../.. python contragent_eval/run_eval_dataflow.py gpt-4o-2024-05-13
# classifier row (2.2%): flag tool outputs with AgentDojo's classifier (needs torch and transformers), then use the flags
PYTHONPATH=../.. python contragent_eval/pi_detector_taint.py runs/gpt-4o-2024-05-13 --attacks important_instructions --out contragent_eval/pi_flags.json
CG_TAINT=contragent_eval/pi_flags.json CG_ATTACKS=important_instructions CG_MODE=taintlib PYTHONPATH=../.. python contragent_eval/run_eval_dataflow.py gpt-4o-2024-05-13
# Figure 5 (all attack classes, 22 models)
CG_MODE=strict   PYTHONPATH=../.. python contragent_eval/run_eval_dataflow.py <model> ...
CG_MODE=taintlib PYTHONPATH=../.. python contragent_eval/run_eval_dataflow.py <model> ...
# the raw library without the data-flow gate
PYTHONPATH=../.. python contragent_eval/run_eval.py gpt-4o-2024-05-13
```

`CG_MODE` selects the provenance rule of the data-flow gate in
`run_eval_dataflow.py` (`strict`, `notuser`, `taint`, `taintlib`; the module
docstring defines them). `strict` is the no-detector row. `taintlib` adds the
full library, gated on the presence of an untrusted span, and is the
trace-learned row; the span is the recorded injection text unless `CG_TAINT`
names a flags file written by `pi_detector_taint.py`. The no-defense row is
the `baseASR` column of either script.
