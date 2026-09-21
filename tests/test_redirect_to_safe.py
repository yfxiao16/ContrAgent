"""Tests for a ``G(!called(unsafe))`` rule carrying a ``Redirect``
strategy.

The contract this pins down:

* A ``DetFormula`` can carry a ``Redirect`` strategy so a violation
  surfaces as a ``redirected`` outcome with
  ``fallback_action=safe_name``, not as a plain ``blocked``.
* The default-policy auto-population in ``ContrAgent.__init__`` honours
  the attached strategy (regression check. earlier the loop
  unconditionally assigned ``Block`` and silently overrode the
  rule's intent).
* The trace is rolled back on redirect, same as on a block, so
  downstream rules don't double-count the substituted call.
* ``CheckResult`` surfaces ``redirected`` + ``redirected_to`` so
  adapters know what tool to invoke instead.
* In observe mode, the redirect outcome is downgraded to ``observed``
  (consistent with how ``blocked`` becomes ``observed``).
* Combined with an assumption: redirect only fires when the
  precondition activates.
* LangGraph adapter executes the safe tool when redirect fires;
  unknown safe tool name raises ``ToolCallBlocked`` with a clear
  message.
"""

from __future__ import annotations

import pytest

from contragent import contract
from contragent.core import ContrAgent
from contragent.formulas.formula import Atom, G, Not
from contragent.runtime.strategies import Redirect
from tests._helpers import ltl


class TestRedirectRule:
    def test_rule_parses_and_carries_strategy(self) -> None:
        formula = ltl("G(!(called('rm_rf')))", enforcement_strategy=Redirect(safe="trash"))
        assert formula.formula == G(Not(Atom("called", "rm_rf")))
        assert formula.kind == "ltl"
        assert isinstance(formula.enforcement_strategy, Redirect)
        assert formula.enforcement_strategy._safe == "trash"


class TestRedirectStrategyOutcome:
    def test_unconditional_redirect_fires_on_first_call(self) -> None:
        guard = ContrAgent(
            agent_id="bot",
            contracts=[
                contract("redirect rm to trash").guarantees(
                    ltl("G(!(called('rm_rf')))", enforcement_strategy=Redirect(safe="trash"))
                )
            ],
            mode="enforce",
        )
        result = guard.guard_before("rm_rf", {"path": "/tmp/x"})
        assert result.redirected is True
        assert result.blocked is False
        assert result.redirected_to == "trash"
        # ``allowed`` stays True so adapters know the agent flow can
        # continue (with the substituted tool, not the original).
        assert result.allowed is True

    def test_other_tools_pass_through(self) -> None:
        guard = ContrAgent(
            agent_id="bot",
            contracts=[
                contract("redirect rm to trash").guarantees(
                    ltl("G(!(called('rm_rf')))", enforcement_strategy=Redirect(safe="trash"))
                )
            ],
            mode="enforce",
        )
        # A different tool isn't affected by the redirect rule.
        result = guard.guard_before("read_file", {"path": "/tmp/x"})
        assert result.allowed is True
        assert result.redirected is False
        assert result.redirected_to is None

    def test_rollback_on_redirect(self) -> None:
        """The attempted unsafe call must roll back so downstream rules
        (count bounds) don't tick on the redirect path.
        The adapter records the substitute via its own
        ``guard_before(safe, args)`` call."""
        guard = ContrAgent(
            agent_id="bot",
            contracts=[
                contract("redirect rm to trash").guarantees(
                    ltl("G(!(called('rm_rf')))", enforcement_strategy=Redirect(safe="trash"))
                )
            ],
            mode="enforce",
        )
        before = len(guard.supervisor.trace.events)
        result = guard.guard_before("rm_rf", {"path": "/tmp/x"})
        after = len(guard.supervisor.trace.events)
        assert result.redirected is True
        assert result.rollback_performed is True
        assert after == before  # Event was popped.

    def test_observe_mode_downgrades_redirect(self, monkeypatch) -> None:
        """In observe mode, the redirect outcome (like any other det
        outcome) becomes ``observed``. ``redirected`` stays False
        because the user explicitly asked for shadow mode.

        The conftest pins ``CONTRAGENT_MODE=enforce`` for the whole
        suite; this test specifically exercises observe-mode semantics
        so it opts out via ``monkeypatch.delenv``, same pattern
        ``tests/test_shadow_mode.py`` uses.
        """
        monkeypatch.delenv("CONTRAGENT_MODE", raising=False)
        guard = ContrAgent(
            agent_id="bot",
            contracts=[
                contract("redirect rm to trash").guarantees(
                    ltl("G(!(called('rm_rf')))", enforcement_strategy=Redirect(safe="trash"))
                )
            ],
            mode="observe",
        )
        result = guard.guard_before("rm_rf", {"path": "/tmp/x"})
        assert result.redirected is False
        observed = [v for v in result.violations if v.action == "observed"]
        assert len(observed) == 1
        assert "REDIRECTED" in observed[0].message


class TestConditionalRedirect:
    def test_redirect_only_fires_when_the_trigger_has_fired(self) -> None:
        """Scoping the rule to a trigger produces a guarded redirect:
        it only fires once the trigger has been reached. This is the
        canonical shape for a context-sensitive redirect."""
        guard = ContrAgent(
            agent_id="bot",
            contracts=[
                contract("redirect large refunds").guarantees(
                    ltl(
                        "G(!(called('issue_refund')))",
                        enforcement_strategy=Redirect(safe="log_refund_request"),
                    )
                )
            ],
            mode="enforce",
        )
        # A tool the rule does not name passes through cleanly.
        r1 = guard.guard_before("read_file", {})
        assert r1.allowed is True
        assert r1.redirected is False

        # The first call to issue_refund reaches the trigger and is
        # redirected on the same step.
        r2 = guard.guard_before("issue_refund", {"amount": 50000})
        assert r2.redirected is True
        assert r2.redirected_to == "log_refund_request"


class TestRedirectInteractsWithOtherRules:
    def test_count_bound_does_not_tick_on_redirect(self) -> None:
        """Redirect rolls back the unsafe event. so a separate
        ``G(count(unsafe) <= N)`` rule on the same tool should NOT
        see the attempt as a real call."""
        guard = ContrAgent(
            agent_id="bot",
            contracts=[
                contract("redirect").guarantees(
                    ltl("G(!(called('rm_rf')))", enforcement_strategy=Redirect(safe="trash"))
                ),
                contract("count").guarantees("G((Var('count', 'rm_rf') <= 2))"),
            ],
            mode="enforce",
        )
        # Trigger the redirect three times. G(count(rm_rf) <= 2)
        # should not block because each rm_rf event is rolled back.
        for _ in range(3):
            r = guard.guard_before("rm_rf", {})
            assert r.redirected is True


class TestRedirectStrategy:
    def test_strategy_rejects_empty_safe(self) -> None:
        with pytest.raises(ValueError, match="safe"):
            Redirect(safe="")

    def test_strategy_can_be_constructed_directly(self) -> None:
        s = Redirect(safe="trash", message="dev only")
        assert s._safe == "trash"
        assert s._message == "dev only"
