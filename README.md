# ContrAgent

**Symbolic temporal supervision of LLM agents using assume-guarantee contracts.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18128-b31b1b.svg)](https://arxiv.org/abs/2609.18128)

Write rules about what an agent may call, and when. ContrAgent checks
every tool call against them before it runs, in microseconds, with no
model in the loop.

The rules are assume-guarantee contracts: pairs of ALTL<sub>f</sub>
formulas (linear temporal logic on finite traces with linear arithmetic)
over the agent's tool-call trace. Each contract compiles to a
deterministic finite automaton that serves two roles. **Online**, it
gates each tool call before execution and blocks, redirects, or
escalates a violating call. **Offline**, it replays a recorded trace and
returns a verdict with the first violating event. A library is
independent of the agent's model, transfers across agents that share a
tool interface, and is checked for conflicts when loaded.

![ContrAgent: contracts are compiled once into DFA monitors, which gate tool calls online and replay recorded traces offline](docs/figures/framework.png)

## Installation

```bash
pip install -e .              # runtime and command line; Python 3.10+
pip install -e ".[dev]"       # plus pytest, ruff, and z3
```

## Quick start

A library is a YAML file. Each contract has a guarantee `G`, what the
agent must keep, and optionally an assumption `A`, what the environment
must keep. A guarantee can order calls, bound a count, constrain an
argument, or demand a follow-up:

```yaml
version: "1"
agents:
  "*":
    contracts:
      - desc: "identity must be verified before funds move"
        G: {ltl: "(!(called('transfer_funds')) U called('verify_identity')) | G(!(called('transfer_funds')))"}
      - desc: "at most three transfers per session"
        G: {ltl: "G((Var('count', 'transfer_funds') <= 3))"}
      - desc: "every transfer is eventually receipted"
        G: {ltl: "G((called('transfer_funds') -> F(called('send_receipt'))))"}
      - desc: "file reads carry no credential"
        A: {ltl: "G(!(output_has('read_file', 'BEGIN PRIVATE KEY')))"}
        G: {ltl: "G((called('send_email') -> called('read_file')))"}
```

Gate the agent's calls with it:

```python
from contragent import ContrAgent

guard = ContrAgent(agent_id="bank_agent", config="bank.yaml")

result = guard.guard_before("transfer_funds", {"amount": 500})
print(result.blocked)     # True
print(result.feedback)    # The action `transfer_funds` was rejected by policy:
                          # identity must be verified before funds move. Choose a different approach.

guard.guard_before("verify_identity", {})
guard.guard_after("verify_identity", {"ok": True})
print(guard.guard_before("transfer_funds", {"amount": 500}).blocked)   # False

guard.finish_session()    # reports eventualities still owed, such as the receipt
```

A blocked call never reaches the tool; `result.feedback` goes back to the
model as the tool result. A tool result that would falsify an assumption
is suppressed the same way: withheld from the model, and the session
state does not advance on it.

The same library checks a recorded trace from the command line:

```bash
contragent replay trace.json --config bank.yaml
```

[`examples/`](examples/) has this program, a supervised tool-calling
loop, and traces to replay; none needs a model or credentials.

## Documentation

- [Authoring a contract library](docs/authoring.md): the four rule
  shapes, weak versus strong until, which side a predicate belongs on,
  arguments, the checks to run before an agent runs under a library,
  enforcement actions, and the Python form.
- [Interaction predicates](docs/predicates.md): the vocabulary formulas
  are written over, and how each predicate is fed.
- [Conflict check](docs/conflict-check.md): optional solvers for the
  load-time library check.
- [CHASE export](docs/chase.md): contract algebra and model checking on
  an exported library.

## Offline evaluation

```bash
contragent eval traces/ --config contragent/contracts/benchmark/rjudge.yaml --agent "*"
contragent replay trace.json --config contragent/contracts/sopbench/bank.yaml
contragent conflicts --config contragent/contracts/sopbench/hotel.yaml
```

`eval` replays a directory of `safe_*.json` / `unsafe_*.json` traces and
reports precision, recall, and false-positive rate per contract;
`replay` prints the verdict and the first violating event of one trace;
`conflicts` runs the library conflict check.

## Experiments

Both roles are evaluated on four benchmarks with the libraries under
`contragent/contracts/`. Success and safety are in percent; ASR is the
attack success rate.

| Benchmark | Role | Result |
|---|---|---|
| SOPBench (7 domains, gemini-2.5-flash) | online enforcement | success / safety 91 / 32 unguarded, 64 / 94 with the procedure in the prompt, 24 / 98 with an LLM judge, **90 / 98 with ContrAgent** at +0.14 s per task |
| AgentDojo (gpt-4o, `important_instructions`) | online enforcement | ASR 47.7% → **11.1%** with no injection detector and **0.79%** with the recorded injection as the untrusted span, utility unchanged at 72.6%, 0.16 ms per call |
| R-Judge (571 records) | offline evaluation | precision **97.0%**, recall **87.0%**, F₁ 91.8%; the misses are semantic, not procedural |
| tau²-bench (4,464 traces) | offline evaluation | the strongest model passes 25–60% of tasks by outcome but **0%** without a procedural violation, decided at 0.83 ms per call |

Each number's record, with the command that produced it and the summary
file behind it, is under [`experiments/`](experiments/). Third-party
datasets are not redistributed; [`benchmarks/README.md`](benchmarks/README.md)
says where to place each one and how to rerun.

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

Apache-2.0. Copyright 2026 Yifeng Xiao. See [`LICENSE`](LICENSE),
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow.
