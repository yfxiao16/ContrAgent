# ContrAgent

Contract-based deterministic trajectory supervision of LLM agents.

ContrAgent specifies what a tool-using agent may do as assume-guarantee
contracts whose assumptions and guarantees are formulas of linear temporal
logic on finite traces, extended with linear arithmetic (ALTL<sub>f</sub>),
over a fixed vocabulary of interaction predicates evaluated on the agent's
tool-call trace. Each contract compiles to a deterministic finite automaton.
The same automata serve two roles:

* **online**, they gate each tool call before it executes (block, redirect
  to a safe alternative, or escalate to a human), with no model call on the
  hot path;
* **offline**, they replay a recorded trace and return a deterministic,
  reproducible verdict with the first violating event and the violated
  contract.

A contract library is independent of the agent's model and transfers across
agents that share a tool interface. Loading a library runs a conflict check
(minimal unsatisfiable core, then joint satisfiability of the core's
assumptions) so that no two contracts that can be active together impose
guarantees that cannot be met jointly.

This repository is the reference implementation accompanying the paper
*Long-Horizon Symbolic Supervision of LLM Agents Using Contracts*.

## Install

```bash
pip install -e ".[dev]"          # core + test tooling (z3 for the exact numeric check)
pip install -e ".[llm]"          # model providers for the formulation pipeline
```

Python 3.10 or later. The runtime has two dependencies (`pyyaml`, `click`).

## Writing contracts

A library is a YAML file. Each contract has a guarantee `G` and an optional
assumption `A`, written as formulas over the interaction predicates:

```yaml
version: "1"
agents:
  "*":
    contracts:
      - desc: "identity must be verified before funds move"
        A: {ltl: "F(called('transfer_funds'))"}
        G: {ltl: "(!(called('transfer_funds')) U called('verify_identity')) | G(!(called('transfer_funds')))"}
      - desc: "at most three bill payments per session"
        G: {ltl: "G((Var('count', 'pay_bill') <= 3))"}
      - desc: "no shell command may delete recursively"
        G: {ltl: "G((called('bash') -> !(arg_field_has('bash', 'command', 'rm -rf'))))"}
```

Temporal operators are `G` (always), `F` (eventually), `X` (next), `U`
(until); Boolean connectives are `&`, `|`, `!`, `->`. The prefix spelling
`G(Implies(called(a), F(called(b))))` is accepted as well. A natural-language
requirement (`nl:`) is lifted to a formula by the formulation pipeline when
an `extractor:` section names a model.

### Interaction predicates

| Paper | Formula spelling | Meaning |
|---|---|---|
| Call(T) | `called(T)`, `called_with(T, p)` | tool T is invoked (with arguments matching p) |
| ArgHas(T,f,p) | `arg_field_has(T, f, p)` | argument f of T matches pattern p |
| Path(T,P) | `arg_paths_within(T, P, ...)` | T's file paths lie within P |
| Subset(f,S) | `Subset(ArgValue(T, f), S)` | values in field f lie within set S |
| OutHas(T,p) | `output_has(T, p)` | result of T matches p |
| Said(p), In(p) | `llm_said(p)`, `prompt_contains(p)` | model output / input matches p |
| Match(f,k) | `Eq(ArgValue(T, f), CtxValue(k))` | argument field f equals context value k |
| Ctx(k,v) | `ctx(k, v)` | context key k holds value v |
| Flow(s,d) | `flow(s, d)` | data from source s reaches sink d |
| Has(f) | `contains(f)` | a produced value contains field f |
| Perm(P) | `perm(P)` | caller holds permission P |
| Cnt(T) | `Var('count', T)` | number of T calls so far |
| Run(T) | `Var('consecutive_count', T)` | length of the current run of T |
| Num(T,f) | `Var('arg_numeric', T, f)` | numeric value of argument field f |
| Len(T,f) | `ArgLength(T, f)` | character length of argument field f |
| InLen, Chars | `Var('context_length')`, `Var('char_count')` | length of the model input / response |
| Tok | `Var('token_count')` | cumulative tokens consumed |
| Depth | `Var('delegation_depth')` | agent-delegation depth |
| Since(e) | `Var('time_since', e)` | time elapsed since predicate e held |

Numeric quantities are compared with `<=`, `<`, `>=`, `>`, `==` against
constants or other quantities.

## Online supervision

```python
from contragent import ContrAgent

guard = ContrAgent(agent_id="bank_agent", config="contragent/contracts/sopbench/bank.yaml")

result = guard.guard_before("transfer_funds", {"amount": 500})
if result.blocked:
    agent_feedback = result.feedback       # returned to the model instead of the tool result
elif result.redirected:
    call(result.redirected_to)
else:
    out = call("transfer_funds", amount=500)
    guard.guard_after("transfer_funds", out)  # for guarantees over tool results

guard.finish_session()                        # decides the pending eventualities
```

Data-flow and context predicates are fed through `observe_data_write`,
`observe_data_read`, `observe_delegation`, `observe_context`, and
`observe_llm_call`. A per-contract enforcement strategy is chosen with
`policy={"<contract desc>": Redirect("safe_tool")}`; the default is `Block`,
and a failed assumption reports through `Escalate` without gating the call.

## Offline evaluation

```bash
contragent eval traces/ --config contragent/contracts/benchmark/rjudge.yaml --agent "*"
contragent replay trace.json --config contragent/contracts/sopbench/bank.yaml
contragent conflicts --config contragent/contracts/sopbench/hotel.yaml
```

`eval` replays a directory of `safe_*.json` / `unsafe_*.json` traces and
reports precision, recall, and false-positive rate per contract; `replay`
prints the end-of-trace verdict, the first violating event, and the
violated contracts of one trace; `conflicts` runs the library conflict
check. The trace format is a JSON object with an `events` list, each event
carrying `ts`, `agent`, `type` (`tool_call`, `tool_output`, `data_read`,
`data_write`, `message`, `context_update`, `llm_request`, `llm_response`),
`tool`, `args`, and `content`.

## Experiments

The paper evaluates both roles on four benchmarks. Contract libraries ship
under `contragent/contracts/`; the harnesses that convert each benchmark's
data to traces and score the results are under `benchmarks/`; the
experiment records are under `experiments/`.

| Benchmark | Role | Library | Harness |
|---|---|---|---|
| SOPBench | enforcement | `contracts/sopbench/*.yaml` | `benchmarks/SOPBench/contragent_eval/` |
| AgentDojo | enforcement | `contracts/benchmark/agentdojo.yaml` | `benchmarks/AgentDojo/contragent_eval/` |
| R-Judge | evaluation | `contracts/benchmark/rjudge.yaml` | `benchmarks/R-Judge/contragent_eval/` |
| tau²-bench | evaluation | `contracts/benchmark/tau2_bench.yaml` | `benchmarks/tau2/contragent_eval/` |

Third-party datasets are not redistributed; each harness README says where
to obtain them and how to convert them. `benchmarks/latency_bench.py`
reproduces the hot-path latency table and `benchmarks/temporal_expressiveness.py`
the temporal-expressiveness table of the appendix.

## Layout

```
contragent/
  formulas/      ALTLf AST, parsers, pointwise evaluator, DFA monitor, LTLf satisfiability, SMT theory
  tracer/        grounding of trace events into predicate valuations
  models/        Contract, Agent, System, Trace, spans
  runtime/       Supervisor (online), TraceVerifier (offline), enforcement strategies
  analysis/      library conflict check (MUC + joint satisfiability; optional mus2muc backend)
  generation/    contract formulation from natural language and policy documents
  discovery/     loaders for requirement artifacts (documents, traces)
  contracts/     shipped contract libraries
  config.py      library files, includes, compilation
  core.py        ContrAgent facade
  cli.py         eval / replay / conflicts
```

## Tests

```bash
pytest
```

## License

BSD 3-Clause. See `LICENSE`.
