"""Enforcement actions taken when a contract fails at runtime.

A violated guarantee is routed to the strategy attached to the contract
(or to the default ``Block``). Three actions are available:

* ``Block`` rejects the call and tells the agent why (the default);
* ``Redirect`` substitutes a safe alternative tool call;
* ``Escalate`` pauses the call for a human decision and fires the
  configured notifiers.

A failed *assumption* is not routed to a strategy. An assumption states
what the environment is required to keep, so the supervisor suppresses
the offending environment event, and it never reaches the agent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from contragent.models.result import Violation


@dataclass
class ActionContext:
    """The action being checked and where in the trace it occurs."""

    agent_id: str
    action: str
    trace_length: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class EnforcementResult:
    """Outcome of one contract check on one action.

    ``action`` is the decision the integration must honour:
    ``blocked`` (do not run the tool, show ``agent_msg``),
    ``redirected`` (run ``fallback_action`` instead), ``escalated``
    (hold for a human; the call is not gated by default), ``suppressed``
    (an environment event that would falsify an assumption; it is
    withheld from the agent), ``allowed`` (no violation), or
    ``observed`` (a would-be decision downgraded to a log entry because
    the supervisor runs in flag mode).
    """

    action: Literal["blocked", "escalated", "redirected", "suppressed", "allowed", "observed"]
    message: str
    fallback_action: Any | None = None
    rule_id: str = ""
    agent_msg: str = ""
    alternatives: list[str] = field(default_factory=list)


def _rule_id_from_violation(violation: Violation) -> str:
    """Stable identifier of the contract that fired."""
    if violation.desc:
        return violation.desc
    formula = getattr(violation, "formula", None)
    kind = getattr(formula, "kind", "") if formula is not None else ""
    return kind or violation.kind


class OutcomeBuilder:
    """Builds the structured result for each enforcement action."""

    @staticmethod
    def for_block(
        violation: Violation,
        context: ActionContext,
        alternatives: list[str] | None = None,
    ) -> EnforcementResult:
        rule = _rule_id_from_violation(violation)
        desc = violation.desc or violation.kind
        message = f"BLOCKED: {context.agent_id}.{context.action}. contract violated: {desc}"
        agent_msg = (
            f"The action `{context.action}` was rejected by policy "
            f"({rule}): {desc}. Choose a different approach."
        )
        return EnforcementResult(
            action="blocked",
            message=message,
            rule_id=rule,
            agent_msg=agent_msg,
            alternatives=list(alternatives or []),
        )

    @staticmethod
    def for_escalate(
        violation: Violation,
        context: ActionContext,
        reason: str = "",
    ) -> EnforcementResult:
        rule = _rule_id_from_violation(violation)
        why = reason or violation.desc or "contract violation"
        message = f"ESCALATED: {context.agent_id}.{context.action}. awaiting human approval: {why}"
        agent_msg = (
            f"The action `{context.action}` is paused awaiting human "
            f"approval ({rule}). Wait for the approval signal."
        )
        return EnforcementResult(
            action="escalated",
            message=message,
            rule_id=rule,
            agent_msg=agent_msg,
        )

    @staticmethod
    def for_redirect(
        violation: Violation,
        context: ActionContext,
        safe: str,
        message: str = "",
    ) -> EnforcementResult:
        rule = _rule_id_from_violation(violation)
        msg = f"REDIRECTED: {context.agent_id}.{context.action} -> {safe}" + (
            f" ({message})" if message else ""
        )
        return EnforcementResult(
            action="redirected",
            message=msg,
            rule_id=rule,
            fallback_action=safe,
            agent_msg="",
            alternatives=[safe],
        )


@runtime_checkable
class EnforcementStrategy(Protocol):
    def enforce(self, violation: Violation, context: ActionContext) -> EnforcementResult: ...


class Block:
    """Reject the call (the default strategy)."""

    def enforce(self, violation: Violation, context: ActionContext) -> EnforcementResult:
        return OutcomeBuilder.for_block(violation, context)


class Escalate:
    """Hold the call for a human and fire the notifiers.

    ``notify`` is a callable or list of callables invoked with
    ``(violation, context, reason)``. Notifier exceptions are caught and
    reported as warnings so an unreachable pager never crashes the agent
    loop. The result is ``escalated``; whether the tool still runs is the
    integration's decision (``ContrAgent.guard_before`` does not gate on
    it, see :class:`~contragent.core.CheckResult`).
    """

    def __init__(
        self,
        reason: str = "",
        notify: Callable | list[Callable] | None = None,
    ) -> None:
        self._reason = reason
        if notify is None:
            self._notifiers: list[Callable] = []
        elif callable(notify):
            self._notifiers = [notify]
        elif isinstance(notify, list) and all(callable(n) for n in notify):
            self._notifiers = list(notify)
        else:
            raise TypeError("Escalate.notify must be a callable, list of callables, or None.")

    def enforce(self, violation: Violation, context: ActionContext) -> EnforcementResult:
        import warnings

        for fn in self._notifiers:
            try:
                fn(violation, context, self._reason)
            except Exception as exc:  # noqa: BLE001 (notifier sandbox)
                warnings.warn(
                    f"Escalate notifier {getattr(fn, '__name__', repr(fn))} "
                    f"raised {type(exc).__name__}: {exc}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return OutcomeBuilder.for_escalate(violation, context, reason=self._reason)


class Redirect:
    """Substitute a pre-approved safe tool for the violating call."""

    def __init__(self, safe: str, message: str = "") -> None:
        if not isinstance(safe, str) or not safe.strip():
            raise ValueError("Redirect: 'safe' must be a non-empty tool name.")
        self._safe = safe
        self._message = message

    def enforce(self, violation: Violation, context: ActionContext) -> EnforcementResult:
        return OutcomeBuilder.for_redirect(
            violation, context, safe=self._safe, message=self._message
        )


__all__ = [
    "ActionContext",
    "EnforcementResult",
    "EnforcementStrategy",
    "Block",
    "Escalate",
    "Redirect",
    "OutcomeBuilder",
]
