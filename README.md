# Long-Horizon Symbolic Supervision of LLM Agents Using Contracts

**ContrAgent** is the reference implementation of the paper of the same
name. It supervises a tool-using LLM agent with assume-guarantee contracts:
each contract is a pair of ALTL<sub>f</sub> formulas (linear temporal logic on
finite traces with linear arithmetic) over a fixed vocabulary of interaction
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

## Installation

Python 3.10 or later. The runtime depends only on `pyyaml` and `click`.

```bash
pip install -e .              # runtime and command line
pip install -e ".[dev]"       # plus pytest, ruff, and z3
pip install -e ".[llm]"       # plus model providers for contract formulation
```

## Contracts

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
      - desc: "no recursive deletion from the shell"
        G: {ltl: "G((called('bash') -> !(arg_field_has('bash', 'command', 'rm -rf'))))"}
```

Temporal operators are `G` (always), `F` (eventually), `X` (next), and `U`
(until); connectives are `&`, `|`, `!`, and `->`. The prefix spelling
`G(Implies(called(a), F(called(b))))` is accepted as well. A natural-language
requirement (`nl:`) is lifted to a formula by the formulation pipeline when
an `extractor:` section names a model. Another library is pulled in with
`include: [contragent:sopbench/bank]`.

### Interaction predicates

| Paper | Formula spelling | Meaning |
|---|---|---|
| Call(T) | `called(T)`, `called_with(T, p)` | tool T is invoked (with arguments matching p) |
| ArgHas(T,f,p) | `arg_field_has(T, f, p)` | argument f of T matches pattern p |
| Path(T,P) | `arg_paths_within(T, P, ...)` | T's file paths lie within P |
| Subset(f,S) | `Subset(ArgValue(T, f), S)` | values in field f lie within set S |
| OutHas(T,p) | `output_has(T, p)` | result of T matches pattern p |
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

Numeric quantities are compared with `<=`, `<`, `>=`, `>`, and `==`.

## Online supervision

```python
from contragent import ContrAgent

guard = ContrAgent(agent_id="bank_agent", config="contragent/contracts/sopbench/bank.yaml")

result = guard.guard_before("transfer_funds", {"amount": 500})
if result.blocked:
    reply = result.feedback                 # returned to the model instead of a tool result
elif result.redirected:
    call(result.redirected_to)
else:
    out = call("transfer_funds", amount=500)
    guard.guard_after("transfer_funds", out)  # guarantees over tool results

guard.finish_session()                      # decides the pending eventualities
```

Data-flow and context predicates are fed through `observe_data_write`,
`observe_data_read`, `observe_delegation`, `observe_context`, and
`observe_llm_call`. The enforcement action of a contract is set with
`policy={"<contract desc>": Redirect("safe_tool")}`; the default is `Block`.
A failed assumption is reported through `Escalate` and does not gate the call.

## Offline evaluation

```bash
contragent eval traces/ --config contragent/contracts/benchmark/rjudge.yaml --agent "*"
contragent replay trace.json --config contragent/contracts/sopbench/bank.yaml
contragent conflicts --config contragent/contracts/sopbench/hotel.yaml
```

`eval` replays a directory of `safe_*.json` / `unsafe_*.json` traces and
reports precision, recall, and false-positive rate per contract. `replay`
prints the verdict, the first violating event, and the violated contracts of
one trace. `conflicts` runs the library conflict check. A trace is a JSON
object with an `events` list; each event carries `ts`, `agent`, `type`
(`tool_call`, `data_read`, `data_write`, `message`, `context_update`,
`llm_request`, `llm_response`), `tool`, `args`, and `content`.

## Conflict check and optional solvers

The conflict check treats the library as the conjunction of its contracts,
extracts a minimal unsatisfiable core, and tests whether the assumptions of
that core are jointly satisfiable. It runs with no extra dependencies; two
optional tools refine it.

* **Z3** (`pip install -e ".[smt]"`) makes the numeric consistency filter
  exact. Without it a built-in interval checker is used.
* **mus2muc** enumerates every minimal core instead of the disjoint cores
  found by the built-in search. Install the package with
  `pip install git+https://github.com/ainnoot/mus2muc`, build its patched
  `wasp` solver and an LTL<sub>f</sub> solver (`aaltaf` or `black`) as
  described in its README, and point `CONTRAGENT_MUS2MUC_BIN` at the folder
  holding the binaries. `contragent conflicts --backend mus2muc` then uses it;
  the default `--backend auto` uses it whenever it is available.

## Design-time analysis with CHASE

[CHASE](https://chase-cps.github.io) is a contract-based requirement-engineering
framework with a contract algebra (composition, conjunction, refinement) and
model-checking and synthesis back ends. A ContrAgent library exports to CHASE's
*logics* specification language:

```bash
contragent export-chase --config contragent/contracts/sopbench/bank.yaml -o bank.logics
```

The export grounds the library: every instantiated interaction predicate
becomes a proposition, every quantity an integer variable (saturating counters
with an explicit range), and each contract a `CONTRACT` block with its
`Assumptions` and `Guarantees`. Because CHASE reasons over infinite words, the
default `--semantics finite` applies the LTL<sub>f</sub>-to-LTL translation
with an `alive` proposition; `--semantics infinite` exports the formulas as
written. Contract blocks are numbered (`c1`, `c2`, ...) and carry their
description in the comment above them, since CHASE's console crashes on
contract names longer than eight characters.

With CHASE's Python bindings on `PYTHONPATH` (`pychase` from
`chase-cps/core-library`, `pychase_logicsLang` from `chase-cps/logics_tool`,
both built with pybind11), two more paths open up:

* `contragent.analysis.chase.PychaseTranslator` builds CHASE `Contract`
  objects directly and applies the contract algebra to a library
  (`conjoin` for several contracts on one agent, `compose` for contracts on
  different components, `refines` for the refinement check between two
  contracts), identifying the variables that contracts share;
* `contragent.analysis.chase.ChaseSession` loads an exported `.logics` file
  into the CHASE console and runs `verify` (NuSMV model of a contract) or
  `synthesize`, whose outputs go to nuXmv, slugs, or gr1c.
  For a specification read from a logics file CHASE emits the NuSMV model
  with an empty `VAR` block; `ChaseSession.verify` fills it from the
  declarations of the export (`smv_declarations` does the same for a model
  written by the standalone `logics_tool`), so the model runs in nuXmv as is.

Building the bindings: `chase-cps/logics_tool` ships a parser generated by
ANTLR 4.9.2, so link it against the ANTLR 4.9.2 C++ runtime (the runtime
bundled with `chase-cps/third_party` is 4.5.4 and the mismatch crashes the
parser); with that runtime, replace `ANTLRFileStream input(infile)` in
`LogicsSpecsBuilder.cc` by `ANTLRFileStream input; input.loadFromFile(infile)`.
A `logics_tool` command file terminates each command with a semicolon;
the console API takes the bare command.

## Experiments

The paper evaluates both roles on four benchmarks. Libraries ship under
`contragent/contracts/`, the harnesses that convert each benchmark's data and
score the results under `benchmarks/`, and the experiment records under
`experiments/`. Third-party datasets are not redistributed; each harness says
where to obtain them.

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
```

## Tests

```bash
pytest
```

## License

BSD 3-Clause. See `LICENSE`.
