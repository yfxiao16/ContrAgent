"""Online supervision of an agent's tool-call trace.

The :class:`Supervisor` holds the compiled contract library and the
trace of the running session. Every proposed action is appended to the
trace, every interaction predicate is evaluated at the new event, and
each contract's monitor is advanced on the resulting valuation. The
system verdict reads the per-contract valuations twice: their meet
gives the agent's side, so a contract whose guarantee fails contributes
FAIL and routes the action to its enforcement strategy, and their join
gives the environment's side, so a contract whose assumption fails
contributes IDLE and suppresses the offending event without blocking
the agent.

A contract's assumption is maintained from the environment side: an
event that would falsify it is suppressed rather than left to turn the
contract IDLE. ``mode`` selects
what the supervisor does with a decision, ``"gate"`` acting on it and
``"flag"`` recording it without gating the agent.
"""

from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from contragent.models.contract import Contract
from contragent.models.result import Violation
from contragent.models.spans import AgentTurnSpan, SpanCollector
from contragent.models.system import System
from contragent.models.trace import Event, Trace
from contragent.runtime.perf import CheckTimer, PerformanceTracker
from contragent.runtime.strategies import (
    ActionContext,
    Block,
    EnforcementResult,
    EnforcementStrategy,
)
from contragent.runtime.verifier import TraceVerifier, Verdict

logger = logging.getLogger(__name__)


@dataclass
class SupervisionEvent:
    """One contract check on one action, as seen by callbacks and logs."""

    agent_id: str
    action: str
    constraint_name: str
    result: EnforcementResult


class Supervisor:
    """Advance every contract monitor over the trace and gate each action.

    Args:
        system: The contract library, one :class:`Contract` per entry.
        policy: Optional mapping from a contract's lookup key (its
            description) to the enforcement strategy applied when it
            fails. Unlisted guarantees fall back to the strategy carried
            by the formula, then to :class:`Block`; failed assumptions
            report through :class:`Escalate` without gating the call.
        mode: ``"gate"`` returns the enforcement decision; ``"flag"``
            evaluates identically but downgrades every decision to
            ``observed`` so the agent is never gated.

    Thread safety: the append-evaluate-decide pipeline runs under one
    re-entrant lock, so concurrent callers see a consistent trace.
    """

    def __init__(
        self,
        system: System,
        policy: dict[str, EnforcementStrategy] | None = None,
        mode: str = "gate",
    ) -> None:
        mode = {"enforce": "gate", "observe": "flag"}.get(mode, mode)
        if mode not in ("gate", "flag"):
            raise ValueError(f"mode must be 'gate' or 'flag', got {mode!r}")
        self._system = system
        self._policy = policy or {}
        self._mode = mode
        self._atom_caches: weakref.WeakKeyDictionary[Any, dict[tuple[int, int], float]] = (
            weakref.WeakKeyDictionary()
        )
        self._lock = threading.RLock()
        self._log: list[SupervisionEvent] = []
        self._trace = Trace(events=[])
        self._callbacks: list[Callable[[SupervisionEvent], None]] = []
        self._last_turn_span: AgentTurnSpan | None = None
        self._turn_spans: list[AgentTurnSpan] = []
        self._verifier = TraceVerifier(backend="dfa")
        self._perf_tracker = PerformanceTracker()
        self._dry_run_depth = 0

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def system(self) -> System:
        return self._system

    @property
    def verifier(self) -> TraceVerifier:
        return self._verifier

    @property
    def trace(self) -> Trace:
        return self._trace

    @property
    def performance_tracker(self) -> PerformanceTracker:
        return self._perf_tracker

    @property
    def log(self) -> list[SupervisionEvent]:
        with self._lock:
            return list(self._log)

    @property
    def last_turn_span(self) -> AgentTurnSpan | None:
        return self._last_turn_span

    @property
    def turn_spans(self) -> list[AgentTurnSpan]:
        with self._lock:
            return list(self._turn_spans)

    def import_trace(self, trace: Trace) -> None:
        """Replace the session trace (used to replay a recorded trace)."""
        with self._lock:
            self._trace = trace
            self._verifier.reset()
            self._last_turn_span = None
            self._turn_spans.clear()
            self._atom_caches.clear()

    def rollback_last_event(self) -> bool:
        """Drop the last event, as if the blocked action never happened."""
        with self._lock:
            if not self._trace.events:
                return False
            self._trace.events.pop()
            self._verifier.reset()
            self._atom_caches.clear()
            return True

    def reset(self) -> None:
        with self._lock:
            self._trace = Trace(events=[])
            self._log.clear()
            self._last_turn_span = None
            self._turn_spans.clear()
            self._verifier.reset()
            self._atom_caches.clear()

    # ------------------------------------------------------------------
    # Checking
    # ------------------------------------------------------------------

    def check_action(
        self,
        agent_id: str,
        action: str,
        event_type: str = "tool_call",
        metadata: dict | None = None,
        dry_run: bool = False,
    ) -> list[EnforcementResult]:
        """Append the proposed action and evaluate every contract on it.

        Returns one :class:`EnforcementResult` per violated contract; an
        empty list means the action is allowed. With ``dry_run`` the
        event is still appended (the caller rolls it back) but nothing is
        logged, emitted, or timed.
        """
        with self._lock:
            if dry_run:
                self._dry_run_depth += 1
            try:
                meta = metadata or {}
                event = Event(
                    ts=len(self._trace.events),
                    agent=agent_id,
                    event_type=event_type,
                    tool=action if event_type == "tool_call" else None,
                    key=meta.get("key"),
                    contains=meta.get("contains"),
                    to=meta.get("to"),
                    args=meta.get("args"),
                    content=meta.get("content"),
                )
                self._trace.events.append(event)
                context = ActionContext(
                    agent_id=agent_id,
                    action=action,
                    trace_length=len(self._trace.events),
                    metadata=meta,
                )
                return self._evaluate(agent_id, context, dry_run=dry_run)
            finally:
                if dry_run:
                    self._dry_run_depth -= 1

    def recheck(self, agent_id: str, action: str) -> list[EnforcementResult]:
        """Re-evaluate the contracts on the current trace without a new event.

        Used after a tool result is attached to the last call, so that
        guarantees over tool outputs are decided as soon as the output is
        known.
        """
        with self._lock:
            if not self._trace.events:
                return []
            self._verifier.reset()
            context = ActionContext(
                agent_id=agent_id,
                action=action,
                trace_length=len(self._trace.events),
                metadata={},
            )
            return self._evaluate(agent_id, context, dry_run=False)

    def _evaluate(
        self, agent_id: str, context: ActionContext, *, dry_run: bool
    ) -> list[EnforcementResult]:
        results: list[EnforcementResult] = []
        with SpanCollector(agent_id, context.action) as collector:
            results.extend(self._check_contracts(agent_id, context, collector))
            collector.root.total_contracts_checked = sum(
                1 for c in collector.root.children if c.span_type == "contragent.contract_check"
            )
            collector.root.violations = len(results)
            collector.root.blocked = any(r.action == "blocked" for r in results)
            if results:
                collector.root.status = "violated"
        if not dry_run:
            self._last_turn_span = collector.root
            self._turn_spans.append(collector.root)
        return results

    def _check_contracts(
        self,
        agent_id: str,
        context: ActionContext,
        collector: SpanCollector,
    ) -> list[EnforcementResult]:
        results: list[EnforcementResult] = []
        agents = {c.agent.id: c.agent for c in self._system.contracts}
        self._verifier.set_agents(agents)
        self._verifier.sync_from_contracts(self._trace, self._system.contracts)

        for contract in self._system.contracts:
            if contract.agent.id != agent_id:
                continue
            a_count = len(contract.assumptions)
            g_count = len(contract.guarantees)
            label = contract.desc or f"{contract.agent.id}: {a_count}A/{g_count}G"
            collector.start_contract_check(label)
            tracker = None if self._dry_run_depth > 0 else self._perf_tracker
            with CheckTimer(tracker, label):
                verdict = self._verifier.check_contract(contract)

            assumption_violated = False
            for a_verdict in verdict.assumptions:
                pre_span = collector.start_precondition(a_verdict.desc)
                if a_verdict.holds:
                    collector.finish_span("ok")
                    self._emit_pass(agent_id, context.action, f"assumption: {a_verdict.desc}")
                    continue
                pre_span.result = False
                collector.finish_span("violated")
                results.append(
                    self._handle_assumption_failure(
                        agent_id, context, collector, a_verdict, contract
                    )
                )
                assumption_violated = True
                break
            if assumption_violated:
                collector.finish_span("violated")
                continue

            contract_violated = False
            for g_verdict in verdict.guarantees:
                guar_span = collector.start_guarantee(g_verdict.desc)
                if g_verdict.holds or not g_verdict.fresh:
                    # Not violated, or violated earlier by a previous event
                    # (already reported then); either way this event passes.
                    collector.finish_span("ok")
                    self._emit_pass(agent_id, context.action, g_verdict.desc)
                    continue
                guar_span.result = False
                collector.finish_span("violated")
                results.append(
                    self._handle_guarantee_failure(agent_id, context, collector, g_verdict)
                )
                contract_violated = True
            collector.finish_span("violated" if contract_violated else "ok")
        return results

    # ------------------------------------------------------------------
    # Outcomes
    # ------------------------------------------------------------------

    def _maybe_downgrade(self, result: EnforcementResult) -> EnforcementResult:
        if self._dry_run_depth > 0 or self._mode != "flag":
            return result
        return EnforcementResult(
            action="observed",
            message=f"OBSERVED (would {result.action}): {result.message}",
            fallback_action=result.fallback_action,
            rule_id=result.rule_id,
            agent_msg=result.agent_msg,
            alternatives=list(result.alternatives),
        )

    def _emit(self, event: SupervisionEvent) -> None:
        with self._lock:
            if self._dry_run_depth > 0:
                return
            self._log.append(event)
            callbacks = list(self._callbacks)
        for fn in callbacks:
            fn(event)

    def _emit_pass(self, agent_id: str, action: str, constraint_name: str) -> None:
        self._emit(
            SupervisionEvent(
                agent_id=agent_id,
                action=action,
                constraint_name=constraint_name,
                result=EnforcementResult(action="allowed", message=f"PASSED: {constraint_name}"),
            )
        )

    def _handle_assumption_failure(
        self,
        agent_id: str,
        context: ActionContext,
        collector: SpanCollector,
        a_verdict: Verdict,
        contract: Contract | None = None,
    ) -> EnforcementResult:
        details = (
            f"Assumption violated: {a_verdict.desc}. "
            "The event that falsified it is withheld from the agent."
        )
        violation = Violation(
            agent_id=agent_id,
            formula=a_verdict.formula,
            kind="assumption",
            desc=a_verdict.desc,
            details=details,
        )
        collector.add_violation(kind="assumption", severity="HIGH", evidence=violation.details)
        collector.add_enforcement(strategy="Suppress", result_action="suppressed")
        result = self._maybe_downgrade(
            EnforcementResult(
                action="suppressed",
                message=details,
                rule_id=a_verdict.desc or "",
                agent_msg=(
                    f"The result of {context.action} was withheld: it violates "
                    f"the assumption {a_verdict.desc!r}."
                ),
            )
        )
        self._emit(
            SupervisionEvent(
                agent_id=agent_id,
                action=context.action,
                constraint_name=f"assumption: {a_verdict.desc}",
                result=result,
            )
        )
        return result

    def _handle_guarantee_failure(
        self,
        agent_id: str,
        context: ActionContext,
        collector: SpanCollector,
        g_verdict: Verdict,
    ) -> EnforcementResult:
        violation = Violation(
            agent_id=agent_id,
            formula=g_verdict.formula,
            kind="guarantee",
            desc=g_verdict.desc,
            details=f"Guarantee violated: {g_verdict.desc}",
        )
        strategy = self._policy.get(g_verdict.lookup_key)
        if strategy is None:
            strategy = getattr(g_verdict.formula, "enforcement_strategy", None)
        if strategy is None:
            strategy = Block()
        result = self._maybe_downgrade(strategy.enforce(violation, context))
        collector.add_violation(kind="guarantee", severity="HIGH", evidence=violation.details)
        collector.add_enforcement(strategy=type(strategy).__name__, result_action=result.action)
        self._emit(
            SupervisionEvent(
                agent_id=agent_id,
                action=context.action,
                constraint_name=g_verdict.desc,
                result=result,
            )
        )
        return result


__all__ = ["Supervisor", "SupervisionEvent"]
