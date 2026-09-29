# ContrAgent

**Symbolic temporal supervision of LLM agents using assume-guarantee contracts.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18128-b31b1b.svg)](https://arxiv.org/abs/2609.18128)

Write rules about what an agent may call, and when. ContrAgent checks
every tool call against them before it runs, in microseconds, with no
model in the loop.

ContrAgent is the reference implementation of *Symbolic Temporal Supervision
of LLM Agents Using Contracts*. The rules are assume-guarantee contracts,
each a pair of ALTL<sub>f</sub> formulas (linear temporal logic on finite
traces with linear arithmetic) over a fixed vocabulary of interaction
predicates evaluated on the agent's tool-call trace. Every contract compiles
to a deterministic finite automaton, and the same automata serve two roles:

* **Online**, they gate each tool call before it executes and block,
  redirect, or escalate a violating call. No model is called on this path.
* **Offline**, they replay a recorded trace and return a deterministic
  verdict with the first violating event and the violated contract.

A contract library is independent of the agent's model and transfers across
agents that share a tool interface. Loading a library runs a conflict check
so that no two contracts that can be active together impose guarantees that
cannot be met jointly.

![ContrAgent: contracts are compiled once into DFA monitors, which gate tool calls online and replay recorded traces offline](docs/figures/framework.png)

*Contracts are compiled once into DFA monitors; the same monitors gate tool
calls online (runtime enforcement) and replay recorded traces offline
(evaluation).*

## Installation

Python 3.10 or later. The runtime depends only on `pyyaml` and `click`.

```bash
pip install -e .              # runtime and command line
pip install -e ".[dev]"       # plus pytest, ruff, and z3
pip install -e ".[llm]"       # plus model providers for contract formulation
```

## Quickstart

Put one contract in `bank.yaml`. It says that funds may not move until
identity has been verified:

```yaml
version: "1"
agents:
  "*":
    contracts:
      - desc: "identity must be verified before funds move"
        G: {ltl: "(!(called('transfer_funds')) U called('verify_identity')) | G(!(called('transfer_funds')))"}
```

Load it and gate the agent's calls:

```python
from contragent import ContrAgent

guard = ContrAgent(agent_id="bank_agent", config="bank.yaml")

result = guard.guard_before("transfer_funds", {"amount": 500})
print(result.blocked)     # True
print(result.feedback)    # The action `transfer_funds` was rejected by policy:
                          # identity must be verified before funds move. Choose a different approach.

guard.guard_before("verify_identity", {})
guard.guard_after("verify_identity", {"ok": True})

result = guard.guard_before("transfer_funds", {"amount": 500})
print(result.blocked)     # False

guard.finish_session()    # decides the pending eventualities
```

A blocked call never reaches the tool. `result.feedback` is what goes back to
the model in place of a tool result, so the agent can choose another route.

The same contracts check a recorded trace from the command line:

```bash
contragent replay trace.json --config bank.yaml
```

## Examples

Three runnable programs under [`examples/`](examples/), none needing a
model or credentials: the quickstart above as a script, a supervised
tool-calling loop (`guard_before` in front of each call, `guard_after`
behind it, refusals fed back to the model), and two recorded traces to
replay:

```bash
python3 examples/quickstart.py
python3 examples/agent_loop.py
contragent replay examples/traces/refund_without_check.json --config examples/refund.yaml
```

## Writing contracts

A library is a YAML file. Each contract has a guarantee `G` and an optional
assumption `A`, both formulas over the interaction predicates. Ordering is
only one of the things a guarantee can say; it can also bound a count or
constrain an argument:

```yaml
version: "1"
agents:
  "*":
    contracts:
      - desc: "at most three bill payments per session"
        G: {ltl: "G((Var('count', 'pay_bill') <= 3))"}
      - desc: "no recursive deletion from the shell"
        G: {ltl: "G((called('bash') -> !(arg_field_has('bash', 'command', 'rm -rf'))))"}
      - desc: "every escalation is eventually resolved"
        G: {ltl: "G((called('escalate') -> F(called('resolve'))))"}
```

`G` (always), `F` (eventually), `X` (next), and `U` (until) are the temporal
operators; `&`, `|`, `!`, and `->` the connectives. A natural-language
requirement (`nl:`) is lifted to a formula by the formulation pipeline when
an `extractor:` section names a model. Another library is pulled in with
`include: [contragent:sopbench/bank]`.

The full predicate vocabulary is in
[`docs/predicates.md`](docs/predicates.md), and
[`docs/authoring.md`](docs/authoring.md) is the guide to turning a policy
into a library: the four rule shapes, weak versus strong until, which
side a predicate belongs on, and how to check a library before an agent
runs under it.

### Assumptions

A contract's assumption states what the environment is required to keep, and
the supervisor maintains it rather than only observing it. A tool result that
would falsify the assumption is suppressed: `guard_after` returns
`result.suppressed`, the result is not attached to the trace, and the session
state does not advance on it, and the agent is told why.

```yaml
- desc: file reads carry no credential
  A: {ltl: "G(!(output_has('read_file', 'BEGIN PRIVATE KEY')))"}
  G: {ltl: "G((called('send_email') -> called('read_file')))"}
```

An assumption must be written over environment predicates. A condition on the
agent's own actions belongs in the guarantee instead, as `G(trigger -> ...)`;
see [`docs/predicates.md`](docs/predicates.md#which-side-a-predicate-belongs-on).

### In Python

The fluent helper writes the same contracts in code, for a library that is
generated or lives next to the tools:

```python
from contragent import ContrAgent, contract, parse_repr

no_leak = (
    contract("file reads carry no credential")
    .assume(parse_repr("G(!(output_has('read_file', 'BEGIN PRIVATE KEY')))"))
    .guarantees(parse_repr("G((called('send_email') -> called('read_file')))"))
)
guard = ContrAgent(agent_id="assistant", contracts=[no_leak])
```

## Online supervision

Data-flow and context predicates are fed through `observe_data_write`,
`observe_data_read`, `observe_delegation`, `observe_context`, and
`observe_llm_call`. The enforcement action of a contract is set with
`policy={"<contract desc>": Redirect("safe_tool")}`; the default is `Block`.

`ContrAgent(mode=...)` selects what the supervisor does with a decision:
`gate` (default) acts on it, `flag` records the same decision without gating
the agent.

A call the contracts cannot be evaluated on (a contract reads an argument
the call did not carry) is refused rather than passed; tool names are
compared in a canonical spelling, so an MCP wire name `mcp__server__tool`
answers to `tool`. Both are described in
[docs/predicates.md](docs/predicates.md#when-a-predicate-has-no-value).

## Offline evaluation

```bash
contragent eval traces/ --config contragent/contracts/benchmark/rjudge.yaml --agent "*"
contragent replay trace.json --config contragent/contracts/sopbench/bank.yaml
contragent conflicts --config contragent/contracts/sopbench/hotel.yaml
```

`eval` replays a directory of `safe_*.json` / `unsafe_*.json` traces and
reports precision, recall, and false-positive rate per contract. `replay`
prints the verdict, the first violating event, and the violated contracts of
one trace. `conflicts` runs the library conflict check.

A trace is a JSON object with an `events` list; each event carries `ts`,
`agent`, `type` (`tool_call`, `data_read`, `data_write`, `message`,
`context_update`, `llm_request`, `llm_response`), `tool`, `args`, and
`content`.

## Conflict check

The conflict check treats the library as the conjunction of its contracts,
extracts a minimal unsatisfiable core, and tests whether the assumptions of
that core are jointly satisfiable. It needs no extra dependencies; Z3 makes
its numeric filter exact and mus2muc enumerates every minimal core. Setup
for both is in [`docs/conflict-check.md`](docs/conflict-check.md).

A library also exports to the *logics* language of
[CHASE](https://chase-cps.github.io) for contract algebra and model
checking (`contragent export-chase`); see [`docs/chase.md`](docs/chase.md).

## Experiments

The paper evaluates both roles on four benchmarks. Libraries ship under
`contragent/contracts/`, the harnesses that convert each benchmark's data and
score the results under `benchmarks/`, and the experiment records under
`experiments/`. Third-party datasets are not redistributed;
[`benchmarks/README.md`](benchmarks/README.md) says where to obtain each and
how to rerun it.

| Benchmark | Role | Library | Harness |
|---|---|---|---|
| SOPBench | online enforcement | `contracts/sopbench/*.yaml` | `benchmarks/SOPBench/contragent_eval/` |
| AgentDojo | online enforcement | `contracts/benchmark/agentdojo.yaml` | `benchmarks/AgentDojo/contragent_eval/` |
| R-Judge | offline evaluation | `contracts/benchmark/rjudge.yaml` | `benchmarks/R-Judge/contragent_eval/` |
| tau²-bench | offline evaluation | `contracts/benchmark/tau2_bench.yaml` | `benchmarks/tau2/contragent_eval/` |

`benchmarks/latency_bench.py` and `benchmarks/temporal_expressiveness.py`
reproduce the two appendix tables.

## Layout

```
contragent/
  formulas/     ALTLf syntax, parsers, pointwise evaluator, DFA monitor, LTLf satisfiability
  tracer/       grounding of trace events into predicate valuations
  models/       Contract, Agent, System, Trace
  runtime/      Supervisor (online), TraceVerifier (offline), enforcement actions
  analysis/     library conflict check (built-in and mus2muc backends)
  generation/   contract formulation from natural language and policy documents
  discovery/    loaders for requirement artifacts
  contracts/    shipped contract libraries
  config.py     library files and compilation
  contract.py   fluent Python helper for writing contracts
  eval_runner.py  offline evaluation of trace directories
  core.py       the ContrAgent supervisor
  cli.py        eval, replay, conflicts, export-chase
examples/       runnable programs: quickstart, a supervised agent loop, traces to replay
```

## Tests

```bash
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Citation

```bibtex
@article{xiao2026contragent,
  title   = {Symbolic Temporal Supervision of {LLM} Agents Using Contracts},
  author  = {Xiao, Yifeng and Nuzzo, Pierluigi},
  journal = {arXiv preprint arXiv:2609.18128},
  year    = {2026}
}
```

## License

Apache-2.0. Copyright 2026 Yifeng Xiao. See [`LICENSE`](LICENSE).
