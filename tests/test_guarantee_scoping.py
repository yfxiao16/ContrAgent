"""Tests for scoping a rule to the events that follow a trigger.

A contract's assumption states what the environment is required to
keep, so a rule that should apply only from some agent action onwards
carries that action in the guarantee, as ``G(trigger -> body)``. These
tests pin the resulting semantics against the unscoped form:

* §1 — the unscoped guarantee flags a pre-trigger event, the scoped one
  does not, on the same trace
* §2 — the scoped guarantee still flags a post-trigger event
* §3 — a trigger that never fires leaves the guarantee satisfied
* §4 — two triggers scope from the later of the two
* §5 — YAML round-trip through the config loader
"""

from __future__ import annotations

import textwrap

from contragent.core import ContrAgent
from contragent.formulas.det import DetFormula
from contragent.formulas.formula import Atom, G, Implies, Not
from contragent.models.agent import Agent
from contragent.models.contract import Contract
from contragent.models.trace import Event, Trace
from contragent.runtime.verifier import TraceVerifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _det(formula, desc: str) -> DetFormula:
    return DetFormula(formula=formula, desc=desc, kind="custom")


def _trace_of(*tools: str, agent: str = "test") -> Trace:
    """Build a Trace from a sequence of tool-call event names."""
    events = []
    for i, t in enumerate(tools):
        events.append(
            Event(ts=i, agent=agent, event_type="tool_call", tool=t, args={})
        )
    return Trace(events=events)


def _verdict_for(contract: Contract, trace: Trace):
    v = TraceVerifier()
    v.sync(trace)
    return v.check_contract(contract)


def _unscoped():
    """G(!Q): Q is forbidden anywhere on the trace."""
    return _det(G(Not(Atom("called", "Q"))), "G(!Q)")


def _scoped():
    """G(P -> G(!Q)): Q is forbidden from the first P onwards."""
    return _det(
        G(Implies(Atom("called", "P"), G(Not(Atom("called", "Q"))))),
        "G(P -> G(!Q))",
    )


def _holds(contract: Contract, trace: Trace) -> bool:
    return all(e.holds for e in _verdict_for(contract, trace).guarantees)


# ---------------------------------------------------------------------------
# §1 — the scope is what separates the two forms
# ---------------------------------------------------------------------------


def test_unscoped_guarantee_flags_q_before_p():
    """Without a trigger, a Q anywhere on the trace is a violation."""
    contract = Contract(agent=Agent(id="t"), guarantee=_unscoped())
    assert not _holds(contract, _trace_of("Q", "P"))


def test_scoped_guarantee_does_not_flag_q_before_p():
    """G(P -> G(!Q)) leaves a Q that happened before the first P alone."""
    contract = Contract(agent=Agent(id="t"), guarantee=_scoped())
    assert _holds(contract, _trace_of("Q", "P"))


# ---------------------------------------------------------------------------
# §2 — a post-trigger event is still caught
# ---------------------------------------------------------------------------


def test_scoped_guarantee_flags_q_after_p():
    contract = Contract(agent=Agent(id="t"), guarantee=_scoped())
    assert not _holds(contract, _trace_of("P", "Q"))


def test_scoped_guarantee_e2e_through_the_guard():
    guard = ContrAgent(agent_id="t", contracts=[Contract(agent=Agent(id="t"), guarantee=_scoped())])
    assert guard.guard_before("Q", {}).allowed is True
    guard.guard_before("P", {})
    assert guard.guard_before("Q", {}).allowed is False


# ---------------------------------------------------------------------------
# §3 — a trigger that never fires
# ---------------------------------------------------------------------------


def test_trigger_never_fires_means_no_violation():
    contract = Contract(agent=Agent(id="t"), guarantee=_scoped())
    assert _holds(contract, _trace_of("Q", "Q", "R"))


# ---------------------------------------------------------------------------
# §4 — two triggers scope from the later one
# ---------------------------------------------------------------------------


def _scoped_twice():
    """G(P -> G(R -> G(!Q))): Q is forbidden once both P and R have fired."""
    inner = G(Implies(Atom("called", "R"), G(Not(Atom("called", "Q")))))
    return _det(G(Implies(Atom("called", "P"), inner)), "G(P -> G(R -> G(!Q)))")


def test_two_triggers_scope_from_the_later_one():
    contract = Contract(agent=Agent(id="t"), guarantee=_scoped_twice())
    # P fires, then Q, then R: Q precedes the later trigger, so it is allowed
    assert _holds(contract, _trace_of("P", "Q", "R"))
    # both triggers fire before Q
    assert not _holds(contract, _trace_of("P", "R", "Q"))


def test_one_trigger_missing_means_no_violation():
    contract = Contract(agent=Agent(id="t"), guarantee=_scoped_twice())
    assert _holds(contract, _trace_of("P", "Q", "Q"))


# ---------------------------------------------------------------------------
# §5 — YAML round-trip
# ---------------------------------------------------------------------------


def test_yaml_loads_a_scoped_guarantee(tmp_path):
    from contragent.config import load_config

    path = tmp_path / "lib.yaml"
    path.write_text(
        textwrap.dedent(
            """
            agents:
              "*":
                contracts:
                  - desc: scoped rule
                    G: {ltl: "G((called('P') -> G(!(called('Q')))))"}
                  - desc: global rule
                    G: {ltl: "G(!(called('Q')))"}
            """
        ).strip()
    )
    cfg = load_config(str(path))
    entries = cfg.agents["*"].contracts
    assert len(entries) == 2
    assert {e.desc for e in entries} == {"scoped rule", "global rule"}
    assert all(e.assumption is None for e in entries)
