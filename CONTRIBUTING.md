# Contributing to ContrAgent

Thanks for your interest. This page covers the development setup, the
checks a change must pass, and where different kinds of changes belong.

## Development setup

```bash
pip install -e ".[dev]"      # runtime plus pytest, ruff, and z3
pre-commit install           # optional: runs ruff on every commit
```

Python 3.10 or newer. The runtime depends only on `pyyaml` and `click`;
`z3` is needed for the numeric consistency filter of the conflict check
and `.[llm]` for contract formulation from natural language.

## Before you open a pull request

```bash
ruff check contragent tests benchmarks
ruff format --check contragent tests benchmarks
pytest
```

All three run in CI on Python 3.10 and 3.12. No test needs a model or
credentials, and no test needs benchmark data.

A change to the runtime (`contragent/`) needs a test. The suite is
organized by mechanism (`test_atoms.py`, `test_theory.py`,
`test_redirect_to_safe.py`, `test_enforced_assumption.py`,
`test_fluent_contracts.py`, ...). Put the new test next to the mechanism
it exercises.

## Layout

```
contragent/
  formulas/     ALTLf syntax, parsers, pointwise evaluator, DFA monitor, LTLf satisfiability
  tracer/       grounding of trace events into predicate valuations
  models/       Contract, Agent, System, Trace
  runtime/      Supervisor (online), TraceVerifier (offline), enforcement actions
  analysis/     conflict check, dead-end check, completion verdict, CHASE export
  generation/   contract formulation from natural language
  discovery/    loaders for policy documents and traces
  contracts/    shipped contract libraries
  config.py     library files and compilation
  contract.py   fluent Python helper
  eval_runner.py  offline evaluation of trace directories
  core.py       the ContrAgent supervisor
  cli.py        eval, replay, conflicts, export-chase
benchmarks/     harnesses for the four benchmarks (datasets not redistributed)
examples/       runnable programs and traces
docs/           authoring guide, predicate vocabulary, tooling notes
```

## Where things go

| You want to | Change |
|---|---|
| Add a predicate | `contragent/formulas/` (syntax), `contragent/tracer/grounding.py` (how it is read off the trace), `docs/predicates.md` |
| Change an enforcement action (block, redirect, escalate, suppress) | `contragent/runtime/strategies.py` and `runtime/supervisor.py` |
| Change how a library is loaded or compiled | `contragent/config.py` (YAML) and `contragent/core.py` (`ContrAgent(contracts=...)`) |
| Add a static analysis over a library | `contragent/analysis/` |
| Change contract formulation from text | `contragent/generation/` and `contragent/discovery/` |
| Add or change a shipped library | `contragent/contracts/`, and note in `benchmarks/README.md` which experiment uses it |
| Add a benchmark harness | `benchmarks/<name>/contragent_eval/`, without redistributing the dataset |
| Add a runnable example | `examples/`, scripted so it runs without credentials |

## Style

`ruff` enforces lint and formatting (line length 100). Docstrings say
what a function decides and returns; a verdict is a fact and an
enforcement result is a decision, and the two stay in separate types
(`runtime/verifier.py` vs `runtime/strategies.py`).

## Reporting a bug

Open an issue with the contract library (or the smallest subset that
reproduces it), the sequence of `guard_before` / `guard_after` calls or
the recorded trace, and what verdict you expected. A failing test is the
best report.
