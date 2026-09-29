# Changelog

All notable changes to ContrAgent are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Changed
- License changed from BSD-3-Clause to Apache-2.0.
- A refusal now reports the contract's `desc` ("identity must be
  verified before funds move") instead of echoing the formula. A
  formula without a description of its own inherits the contract's.

### Fixed
- Contracts built with the fluent helper (`contract(...).guarantees(...)`)
  or from raw formulas now flag their unbounded eventualities, so
  `finish_session()` reports an owed `F` obligation for them as it
  already did for YAML contracts.

### Removed
- Dead code, none of it referenced by the package, the tests, the
  harnesses, or ACORN: the deprecated `contragent.formulas.fol` module,
  the `Tracer` class of `contragent.tracer` (grounding stays), the
  duplicate `contracts/sopbench/compile_tree.py` (the SOPBench harness
  keeps its copy), the unused baseline-diff types of `eval_runner`, and
  a dozen unreferenced functions and methods (`config_to_guard_kwargs`,
  `make_contracts`, `render_last_turn`, `register_callback`, ...).

### Added
- `docs/authoring.md`: how to turn a policy into a library (the four
  rule shapes, weak versus strong until, which side a predicate belongs
  on, arguments, and the checks to run before an agent runs under it).
- `tests/test_dead_ends.py` for the static dead-end check whose
  verdicts on the shipped libraries the results ledger reports.
- `examples/`: the README quickstart as a script, a supervised
  tool-calling loop, and two recorded traces for `contragent replay`.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, this changelog; CI runs the
  examples.
- `contragent.analysis.completion` and `contragent.analysis.dead_ends`:
  the completion verdict and the static dead-end check under the names
  ACORN uses.

## [1.0.0] - 2026-09-21

Reference implementation of *Symbolic Temporal Supervision of LLM Agents
Using Contracts* (arXiv:2609.18128): ALTLf contracts over interaction
predicates, DFA monitors for online enforcement (block, redirect,
escalate, suppress) and offline replay, the conflict check with minimal
unsatisfiable cores, contract formulation from natural language, and the
SOPBench, AgentDojo, R-Judge, and tau²-bench harnesses.

[Unreleased]: https://github.com/yfxiao16/ContrAgent/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yfxiao16/ContrAgent/releases/tag/v1.0.0
