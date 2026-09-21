"""Tests for the assumption side of a contract and the suppression path.

The contract these pin down:

* An assumption states what the environment is required to keep, so
  every assumption is maintained the same way; there is no per-contract
  mode to choose between evaluating and enforcing it.
* An assumption written over predicates the agent controls is a
  guarantee in disguise and is rejected at construction.
* A failing assumption suppresses the offending tool result: the
  outcome is ``suppressed``, the agent sees feedback, and the output is
  not attached to the trace, so the session state does not advance.
* In ``flag`` mode the same decision is downgraded to ``observed`` and
  the output stays attached.
* ``gate``/``flag`` are the supervisor modes, with ``enforce``/``observe``
  still accepted as the pre-rename spellings.
"""

from __future__ import annotations

import pytest

from contragent import contract
from contragent.core import ContrAgent
from contragent.formulas.parser import parse_repr
from contragent.models.agent import Agent
from contragent.models.contract import Contract
from contragent.runtime.supervisor import Supervisor


def _no_secret_in_output():
    return parse_repr("G(!(output_has('read_file', 'SECRET')))")


def _guard(mode: str = "gate") -> ContrAgent:
    return ContrAgent(
        contracts=[
            contract("file reads carry no secret")
            .assume(_no_secret_in_output())
            .guarantees(parse_repr("G((called('send_email') -> called('read_file')))")),
        ],
        mode=mode,
    )


class TestDeclaration:
    def test_no_assumption_mode_is_carried(self) -> None:
        c = contract("c").assume(_no_secret_in_output()).guarantees(parse_repr("G(called('a'))"))
        assert "assumption_mode" not in c.to_dict()
        assert not hasattr(
            Contract(agent=Agent(id="a"), guarantee=parse_repr("G(called('a'))")), "assumption_mode"
        )

    def test_environment_assumption_is_accepted(self) -> None:
        contract = Contract(
            agent=Agent(id="a"),
            guarantee=parse_repr("G(called('b'))"),
            assumption=_no_secret_in_output(),
        )
        assert contract.assumption is not None

    def test_agent_controlled_assumption_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="which the agent controls"):
            Contract(
                agent=Agent(id="a"),
                guarantee=parse_repr("G(called('b'))"),
                assumption=parse_repr("F(called('transfer'))"),
            )


class TestSuppression:
    def test_clean_output_passes(self) -> None:
        guard = _guard()
        guard.guard_before("read_file", {"path": "/tmp/x"})
        result = guard.guard_after("read_file", "harmless contents")
        assert not result.suppressed
        assert guard.trace.events[-1].content is not None

    def test_offending_output_is_suppressed(self) -> None:
        guard = _guard()
        guard.guard_before("read_file", {"path": "/tmp/x"})
        before = [e.content for e in guard.trace.events]
        result = guard.guard_after("read_file", "here is a SECRET token")
        assert result.suppressed
        assert not result.allowed
        assert "withheld" in result.feedback
        # the output is not attached, so the state does not advance
        assert [e.content for e in guard.trace.events] == before

    def test_flag_mode_reports_without_suppressing(self) -> None:
        guard = _guard(mode="flag")
        guard.guard_before("read_file", {"path": "/tmp/x"})
        result = guard.guard_after("read_file", "here is a SECRET token")
        assert not result.suppressed
        assert any(r.action == "observed" for r in result.violations)
        assert "SECRET" in (guard.trace.events[-1].content or "")


class TestModeNames:
    def test_gate_and_flag(self) -> None:
        assert ContrAgent(contracts=[], mode="gate").mode == "gate"
        assert ContrAgent(contracts=[], mode="flag").mode == "flag"

    def test_pre_rename_spellings_still_accepted(self) -> None:
        assert ContrAgent(contracts=[], mode="enforce").mode == "gate"
        assert ContrAgent(contracts=[], mode="observe").mode == "flag"
        assert Supervisor.__init__ is not None

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            ContrAgent(contracts=[], mode="whatever")
