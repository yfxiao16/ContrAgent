# ContrAgent x tau2-bench (procedure-correctness eval)

Offline replay of tau2-bench's **shipped** simulation transcripts through
the bundled ContrAgent contract library
(`contragent/contracts/benchmark/tau2_bench.yaml`). No API/LLM calls.

## What tau2-bench ships offline

tau2-bench ships full recorded agent run traces (not just the
environment + policies). They live at:

```
benchmarks/tau2/data/tau2/results/final/<model>_<domain>_<variant>_gpt-4.1-2025-04-14_4trials.json
```

26 result files. Each is a `Results` object with `simulations` = (tasks x
4 trials) transcripts. Each simulation carries the full message list
(assistant/user/tool, with real `tool_calls` names + arguments) and a
`reward_info` block (`reward == 1.0` => tau2 task pass).

Canonical leaderboard cells = 4 models x 3 domains, `default`/`base`
variant (the `telecom-workflow` + `no-user`/`op` ablation files are
extra and excluded by this eval):

| model | retail | airline | telecom |
|---|---:|---:|---:|
| claude-3-7-sonnet | 456 | 200 | 456 |
| gpt-4.1 | 456 | 200 | 456 |
| gpt-4.1-mini | 456 | 200 | 456 |
| o4-mini | 456 | 200 | 456 |

(retail 456 = 114 tasks x 4; airline 200 = 50 x 4; telecom 456 = 114 x 4.)

So no agents need to be run. The environment + per-domain `policy.md`
also ship (`data/tau2/domains/<domain>/`) for live runs, but the recorded
traces are sufficient for offline procedure-violation scoring.

## Files

- `convert.py` — tau2 simulation transcript -> ContrAgent native trace
  (`{"metadata", "events":[{ts,agent,type,tool,args,...}]}`). Emits REAL
  tool names/args only. The one derived signal is the structural
  `same_turn_text_and_tool_call` flag (an assistant message containing
  BOTH user-facing text and a tool call — a direct property of the
  message, not a policy inference).
- `eval_proc.py` — loads the contract bundle (tolerant per-entry
  compile), classifies each contract, replays every sim through the
  honestly-evaluable subset with `TraceVerifier`, and reports per-cell
  procedure-violation fire-rates + `pass^4`/`proc-clean^4`/`joint^4`.
- `proc_eval_results.json` — machine-readable per-cell output.

## Honest-evaluation policy

The 120-contract bundle splits, by atom dependency, into:

- **53 honest** — evaluable from real tool names/args/ordering alone:
  all `pattern` contracts (`must_precede`, `rate_limit`,
  `arg_blacklist`, `arg_allowlist`, `arg_value_range`), pure-LTL
  ordering + post-action-verification (`called`/`X(called)`), and the
  structural `same_turn_text_and_tool_call` protocol flag. **These are
  the only contracts that fire in this eval.**
- **63 need derived ctx** — depend on policy-derived facts
  (`user_authenticated`, `target_order_status`, `*_owner_match`,
  `account_status`, `target_line_status`, ...). Deriving those from a
  transcript means re-implementing each domain policy against the
  environment DB — semantic tagging this eval refuses to do. **Skipped.**
- **4 unparseable Cat-C** — `ArgLength`/`ArgValue` contracts whose infix
  form the public `parse_repr` does not yet accept. **Skipped.**

The 63 ctx-gated contracts are exactly what the private
`procedure_correctness/` env-grounding adapter referenced in
`experiments/tau2.md` supplies. To evaluate them honestly you must
stand up that adapter (replay each tool call against the tau2 env and
read back the resulting DB facts) — not infer them from text.

## Run

```sh
PYTHONPATH=<repo> \
  /opt/homebrew/opt/python@3.13/bin/python3.13 \
  benchmarks/tau2/contragent_eval/eval_proc.py
```

Deterministic (bit-identical across runs), 0 LLM calls.

## Cross-check against experiments/tau2.md

The per-category `output_spec` fire-rates reproduce the write-up
exactly (e.g. retail Claude 98.9%, GPT-4.1 2.0%, mini 36.8%, o4-mini
0.0%; airline 98.0/9.0/46.5/0.0; telecom 99.1/38.4/74.1/0.0), and every
Claude cell is `proc-clean^4 = 0.0%` while `pass^4` stays 25-60%. Cells
that differ from the doc's full-bundle numbers (transition_spec, some
blind-spot rates) differ precisely because the doc's run also evaluates
the 63 ctx-gated contracts via the env-grounding adapter, which this
offline-from-text eval correctly does not fabricate.
