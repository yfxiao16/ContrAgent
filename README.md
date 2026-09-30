# ContrAgent

**Symbolic temporal supervision of LLM agents using assume-guarantee contracts.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18128-b31b1b.svg)](https://arxiv.org/abs/2609.18128)

![ContrAgent: contracts are compiled once into DFA monitors, which gate tool calls online and replay recorded traces offline](docs/figures/framework.png)

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

## Installation

```bash
pip install -e .              # runtime and command line; Python 3.10+
pip install -e ".[dev]"       # plus pytest, ruff, and z3
```

## Quick start

A contract is a rule over the order, count, and arguments of tool calls.
Two from [`examples/bank.yaml`](examples/bank.yaml):

```yaml
- desc: "identity must be verified before funds move"
  G: {ltl: "(!(called('transfer_funds')) U called('verify_identity')) | G(!(called('transfer_funds')))"}
- desc: "no email after a file read"
  G: {ltl: "G((called('read_file') -> G(!(called('send_email')))))"}
```

Both are about the shape of the trace, not its content: the first refuses
a transfer until an identity check has happened, the second refuses any
email once a file has been read, whatever the email says. Wrap your tools
and every call is checked:

```python
from contragent import ContrAgent

guard = ContrAgent(agent_id="bank_agent", config="bank.yaml")

@guard.wrap
def transfer_funds(amount: float, to: str) -> dict:
    ...

transfer_funds(500, "ACME")
# -> "The action `transfer_funds` was rejected by policy: identity must be
#     verified before funds move. Choose a different approach."
```

A refused call never runs; the wrapper returns the refusal text, which is
what goes back to the model as the tool result (`on_block="raise"` raises
instead). `guard.wrap({...})` wraps a whole tool table, and loops that
execute tools elsewhere call the two hooks, `guard_before` and
`guard_after`, themselves. With `verbose=True` the session prints as a
timeline as it happens, and `trace_path=` writes it, at
`finish_session()`, as a trace file that `contragent replay` reads back:

```text
bank.yaml · 5 contracts · agent bank_agent · mode gate
│
⊘ transfer_funds(amount=500, to="ACME")
│   identity must be verified before funds move
│
● verify_identity(user_id="u1")
│
● transfer_funds(amount=500, to="ACME")
│
● read_file(path="id_rsa")
│   ◐ result withheld · no private key reaches the model or leaves in an email
│
⊘ send_email(to="x", body="hi")
│   no email after a file read
│
■ session end · 5 calls · 2 refused · 1 withheld · 1 obligation pending
    ◌ every transfer is eventually receipted
```

The same library replays a recorded trace:

```bash
contragent replay trace.json --config bank.yaml
```

[`examples/`](examples/) has a supervised tool-calling loop and traces to
replay; nothing there needs a model or credentials.

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
