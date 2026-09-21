"""Tests for where a conditional guard lands when an IR is compiled.

The contract these pin down:

* A contract's assumption states what the environment is required to
  keep, so only a guard over environment predicates becomes one.
* A guard over the agent's own calls is folded into the guarantee as
  ``G(guard -> body)``, and no assumption is emitted.
* A rule with no guard is unchanged on both sides.
"""

from __future__ import annotations

from contragent.formulas.formula import G, Implies, agent_controlled_atoms
from contragent.generation.structured_ir import ConstraintIR, compile_ir


def _compile(guard: str | None):
    ir = ConstraintIR(
        subject="agent",
        object="issue_refund",
        relation="bans",
        scope="conditional" if guard else "global",
        guard=guard,
    )
    return compile_ir(ir)


class TestGuardRouting:
    def test_agent_guard_folds_into_the_guarantee(self) -> None:
        result = _compile("called('approve')")
        assert result.error in ("", None)
        assert result.compiled_assumption is None
        formula = result.compiled.formula
        assert isinstance(formula, G)
        assert isinstance(formula.child, Implies)

    def test_environment_guard_becomes_the_assumption(self) -> None:
        result = _compile("output_has('read_file', 'SECRET')")
        assert result.error in ("", None)
        assert result.compiled_assumption is not None
        assert not agent_controlled_atoms(result.compiled_assumption.formula)

    def test_no_guard_leaves_both_sides_alone(self) -> None:
        unguarded = _compile(None)
        assert unguarded.compiled_assumption is None
        guarded = _compile("called('approve')")
        # the guarded rule wraps exactly the unguarded body
        assert guarded.compiled.formula.child.right == unguarded.compiled.formula


class TestAgentControlledAtoms:
    def test_environment_predicates_are_not_agent_controlled(self) -> None:
        from contragent.formulas.parser import parse_repr

        assert not agent_controlled_atoms(
            parse_repr("G(!(output_has('read_file', 'SECRET')))")
        )

    def test_called_is_agent_controlled(self) -> None:
        from contragent.formulas.parser import parse_repr

        assert agent_controlled_atoms(parse_repr("F(called('transfer'))")) == {
            "called"
        }


class TestAssumptionEnforceability:
    """An assumption must be maintainable by suppressing an event.

    Suppression keeps an event from reaching the agent but cannot bring
    one about, so an assumption may not demand an eventuality. Weak
    until, which the grammar spells ``(phi U psi) | G phi``, is a safety
    property and stays allowed.
    """

    @staticmethod
    def _contract(assumption: str):
        from contragent.formulas.parser import parse_repr
        from contragent.models.agent import Agent
        from contragent.models.contract import Contract

        return Contract(
            agent=Agent(id="bot"),
            guarantee=parse_repr("G(called('x'))"),
            assumption=parse_repr(assumption),
        )

    def test_safety_assumption_is_accepted(self) -> None:
        c = self._contract("G(!(output_has('t', 'SECRET')))")
        assert c.assumption is not None

    def test_weak_until_is_accepted(self) -> None:
        c = self._contract(
            "(!(output_has('t','a')) U output_has('t','b')) "
            "| G(!(output_has('t','a')))"
        )
        assert c.assumption is not None

    def test_eventually_is_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="unbounded eventuality"):
            self._contract("F(output_has('t', 'ok'))")

    def test_strong_until_is_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="unbounded eventuality"):
            self._contract("!(output_has('t','a')) U output_has('t','b')")

    def test_response_shape_is_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="unbounded eventuality"):
            self._contract(
                "G((output_has('t','a') -> F(output_has('t','b'))))"
            )
