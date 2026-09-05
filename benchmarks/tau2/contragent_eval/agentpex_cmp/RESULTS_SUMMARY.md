# AgentPex (LLM judge) vs deterministic contract library — tau2-bench procedural eval

Judge model: gemini-2.5-flash (AgentPex official code, via OpenAI-compat endpoint;
reasoning_effort=none + max_completion_tokens=32000 to stop truncation).
GT: 53 tool-data-evaluable contracts, per-sim per-category, verifiable/recomputable.
Sample: 300 sims (25 / (domain x model) cell), 294 judged.

## Accuracy of the LLM judge vs verifiable deterministic labels (tau=65)
| metric          | P    | R    | F1   | notes |
|-----------------|------|------|------|-------|
| ANY-VIOLATION   | 0.77 | 0.29 | 0.42 | acc 0.51; TP53 FP16 FN128 TN97 |
| output_spec     | 0.48 | 0.57 | 0.52 | GT pos 118 (~ same-turn text+tool_call) |
| transition_spec | 0.50 | 0.47 | 0.48 | GT pos 110 |
| forbidden_edges | --   | --   | --   | UNUSABLE: AgentPex emits no forbidden score for airline (22 GT pos = NA) |
| argument_spec / predicted_plan | -- | -- | -- | only 2 GT pos each, not reportable |

Headline: the LLM judge MISSES 128 / 181 (71%) of traces that deterministically
contain a procedural violation; precision moderate (0.77); trace-level agreement
barely above chance (acc 0.51).

## Cost (faithful to AgentPex gpt-5-mini config, full 4464-trace matrix)
deterministic: 0 LLM calls, $0, 33 min, bit-identical across runs.
AgentPex LLM judge: 9 calls/trace, ~$85, ~172 h.

## Variance (Phase 5): BLOCKED
gemini key hit monthly spending cap (HTTP 429 RESOURCE_EXHAUSTED) after the main
run. Re-run with LLM_CACHE_DISABLE=1 to measure run-to-run flip rate once the cap
is raised (ai.studio/billing) or resets.

## Validity notes
- Compared at trace level (any procedural violation) = same granularity as
  AgentPex's own 83% blind-spot statistic; categories bucket slightly differently.
- per-category R uses only sims where AgentPex emitted that category's score;
  any-violation uses get(score,100)=clean default. Tighten before final numbers.
