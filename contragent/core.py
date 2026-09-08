"""The ContrAgent supervisor as seen by an agent integration.

``ContrAgent`` loads a contract library, keeps the session trace, and
exposes the two hooks an integration wires around each tool call:
``guard_before`` (gate the call) and ``guard_after`` (attach the result
and re-check output guarantees). Data-flow and context predicates are
fed through the ``observe_*`` hooks; ``finish_session`` decides the
liveness guarantees once the trace is complete. The same object replays
a recorded trace offline through ``evaluate_trace``.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

from contragent.models.agent import Agent
from contragent.models.contract import Contract
from contragent.models.system import System
from contragent.models.trace import Trace
from contragent.runtime.strategies import EnforcementResult, EnforcementStrategy
from contragent.runtime.supervisor import SupervisionEvent, Supervisor
from contragent.runtime.verifier import Verdict

_VALID_MODES = ("gate", "flag")
# Pre-rename spellings, accepted so existing scripts and configs keep working.
_MODE_ALIASES = {"enforce": "gate", "observe": "flag"}


@dataclass
class CheckResult:
    """What the integration does with the proposed action.

    ``allowed`` is False exactly when some contract blocked the call. An
    ``escalated`` result does not gate the call by itself (the contract's
    assumption did not hold, or a human decision is pending); integrations
    that want to hold on escalation check :attr:`escalated`.
    """

    allowed: bool
    violations: list[EnforcementResult] = field(default_factory=list)
    redirected_to: Any | None = None
    rollback_performed: bool = False

    @property
    def blocked(self) -> bool:
        return any(r.action == "blocked" for r in self.violations)

    @property
    def escalated(self) -> bool:
        return any(r.action == "escalated" for r in self.violations)

    @property
    def suppressed(self) -> bool:
        """True when an environment event was withheld from the agent.

        Set when the event would have falsified an assumption the
        contract declares ``enforced``. The tool output is not attached
        to the trace, so the session state does not advance on it.
        """
        return any(r.action == "suppressed" for r in self.violations)

    @property
    def redirected(self) -> bool:
        return self.redirected_to is not None

    @property
    def stop_original(self) -> bool:
        """True when the original tool call must not run (blocked or redirected)."""
        return self.blocked or self.redirected

    @property
    def feedback(self) -> str:
        """Message to show the agent, empty when the action is allowed."""
        msgs = [r.agent_msg or r.message for r in self.violations if r.action != "allowed"]
        return "\n".join(m for m in msgs if m)


def _resolve_mode(mode: str | None) -> str:
    resolved = mode or os.environ.get("CONTRAGENT_MODE") or "gate"
    resolved = _MODE_ALIASES.get(resolved, resolved)
    if resolved not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {resolved!r}")
    return resolved


class ContrAgent:
    """Contract-based supervisor for one agent.

    Args:
        agent_id: Identifier of the supervised agent (matches the agent
            block in a library file).
        contracts: Inline contracts: :class:`Contract` objects, contract
            mappings (``guarantee`` required, ``assumption``/``desc``/
            ``activate_at`` optional), :class:`ContractBuilder` values,
            formulas, or formula strings (each string is a guarantee).
        config: Path to a library file, alternative to ``contracts``.
        system: A pre-built :class:`System`, alternative to both.
        policy: Mapping from a contract description to the enforcement
            strategy applied when it fails.
        mode: ``"gate"`` acts on every decision; ``"flag"`` records
            the same decisions without gating the agent. Defaults to
            the ``CONTRAGENT_MODE`` environment variable, then ``"gate"``.
        conflict_check: Run the library conflict check at load time and
            raise if the library is not conflict-free.
    """

    def __init__(
        self,
        agent_id: str = "agent",
        contracts: list[Any] | None = None,
        config: str | os.PathLike | None = None,
        system: System | None = None,
        policy: dict[str, EnforcementStrategy] | None = None,
        mode: str | None = None,
        conflict_check: bool = False,
    ) -> None:
        if sum(x is not None for x in (contracts, config, system)) > 1:
            raise ValueError("Pass exactly one of 'contracts', 'config', or 'system'.")
        if config is not None:
            from contragent.config import config_to_system, load_config

            parsed = load_config(config)
            if agent_id == "agent" and agent_id not in parsed.agents:
                if len(parsed.agents) == 1:
                    agent_id = next(iter(parsed.agents))
                elif len(parsed.agents) > 1:
                    raise ValueError(
                        f"Library has several agents {list(parsed.agents)}; pass agent_id."
                    )
            system = config_to_system(parsed)
            if "*" in parsed.agents and agent_id != "*":
                # A wildcard block applies to whichever agent is supervised.
                for c in system.contracts:
                    if c.agent.id == "*":
                        c.agent = Agent(id=agent_id)
        elif system is None:
            system = System(name=agent_id)
            system._contracts = self._build_contracts(Agent(id=agent_id), contracts or [])
        self.agent_id = agent_id
        self._system = system
        self._mode = _resolve_mode(mode)
        self._supervisor = Supervisor(system, policy=policy, mode=self._mode)
        self._lock = threading.RLock()
        self._violations: list[dict] = []
        self._pending_liveness: list[Verdict] | None = None
        if conflict_check:
            import sys

            report = self.check_conflicts()
            if not report.ok:
                # A conflicted library still loads; the report goes to stderr so
                # the operator sees which contracts to repair.
                print(
                    f"warning: contract library is not conflict-free:\n{report.render()}",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_constraint(value: Any) -> Any:
        """Accept a formula, a ``DetFormula``, a formula string, or a list of those."""
        from contragent.config import ConstraintEntry, _compile_ltl
        from contragent.formulas.det import DetFormula
        from contragent.formulas.formula import FormulaMixin

        if isinstance(value, list):
            return [ContrAgent._as_constraint(v) for v in value]
        if isinstance(value, (DetFormula, FormulaMixin)):
            return value
        if isinstance(value, str):
            return _compile_ltl(ConstraintEntry(ltl=value))
        raise TypeError(
            f"Contract constraints must be formulas or formula strings, got {type(value).__name__}"
        )

    @classmethod
    def _build_contracts(cls, agent: Agent, entries: list[Any]) -> list[Contract]:
        out: list[Contract] = []
        for entry in entries:
            to_dict = getattr(entry, "to_dict", None)
            if callable(to_dict) and not isinstance(entry, Contract):
                entry = to_dict()
            if isinstance(entry, Contract):
                out.append(entry)
                continue
            if isinstance(entry, dict):
                if "guarantee" not in entry:
                    raise ValueError(f"Contract entry is missing 'guarantee': {entry!r}")
                out.append(
                    Contract(
                        agent=agent,
                        guarantee=cls._as_constraint(entry["guarantee"]),
                        assumption=(
                            None
                            if entry.get("assumption") is None
                            else cls._as_constraint(entry["assumption"])
                        ),
                        desc=entry.get("desc"),
                        activate_at=entry.get("activate_at"),
                        assumption_mode=entry.get("assumption_mode") or "monitored",
                    )
                )
                continue
            out.append(Contract(agent=agent, guarantee=cls._as_constraint(entry)))
        return out

    # ------------------------------------------------------------------
    # Online hooks
    # ------------------------------------------------------------------

    def guard_before(self, tool_name: str, args: dict | None = None) -> CheckResult:
        """Check the contracts before ``tool_name`` runs.

        The call is appended to the trace and every contract is advanced.
        A blocked or redirected call is rolled back from the trace so it
        does not count as having happened.
        """
        with self._lock:
            results = self._supervisor.check_action(
                agent_id=self.agent_id,
                action=tool_name,
                metadata={"args": args} if args else {},
            )
            return self._finish_check(tool_name, results)

    def guard_after(self, tool_name: str, output: Any) -> CheckResult:
        """Attach the tool result to its call and re-check the contracts.

        When the result would falsify an assumption the contract declares
        ``enforced``, it is suppressed: the attachment is undone, so the
        result never reaches the agent and the session state does not
        advance on it.
        """
        with self._lock:
            before = self._output_content(tool_name)
            self.observe_tool_output(tool_name, output)
            results = self._supervisor.recheck(self.agent_id, tool_name)
            result = CheckResult(
                allowed=not any(r.action == "blocked" for r in results),
                violations=[r for r in results if r.action != "allowed"],
            )
            if result.suppressed and self._mode != "flag":
                self._restore_output_content(tool_name, before)
                result.allowed = False
            self._record(tool_name, result.violations)
            return result

    def _output_content(self, tool_name: str) -> tuple[Any, str | None] | None:
        """The event carrying ``tool_name``'s output and its current content."""
        for ev in reversed(self._supervisor.trace.events):
            if ev.event_type == "tool_call" and ev.tool == tool_name and ev.agent == self.agent_id:
                return (ev, ev.content)
        return None

    def _restore_output_content(
        self, tool_name: str, before: tuple[Any, str | None] | None
    ) -> None:
        """Undo the attachment made by :meth:`observe_tool_output`."""
        if before is None:
            return
        event, content = before
        event.content = content
        self._supervisor.verifier.reset()

    def _finish_check(self, tool_name: str, results: list[EnforcementResult]) -> CheckResult:
        redirected = [r for r in results if r.action == "redirected"]
        result = CheckResult(
            allowed=not any(r.action == "blocked" for r in results),
            violations=[r for r in results if r.action != "allowed"],
            redirected_to=redirected[0].fallback_action if redirected else None,
        )
        if self._mode != "flag" and (result.blocked or result.redirected):
            if self._supervisor.rollback_last_event():
                result.rollback_performed = True
        self._record(tool_name, result.violations)
        return result

    def _record(self, tool_name: str, violations: list[EnforcementResult]) -> None:
        for r in violations:
            self._violations.append(
                {"tool": tool_name, "constraint": r.message, "action": r.action.upper()}
            )

    def observe_tool_output(self, tool_name: str, output: Any) -> None:
        """Attach ``output`` to the most recent call of ``tool_name`` (for OutHas)."""
        trace = self._supervisor.trace
        for ev in reversed(trace.events):
            if ev.event_type == "tool_call" and ev.tool == tool_name and ev.agent == self.agent_id:
                text = str(output)
                ev.content = text if ev.content is None else ev.content + text
                self._supervisor.verifier.reset()
                return
        import warnings

        warnings.warn(
            f"observe_tool_output({tool_name!r}): no preceding call of this tool "
            f"on agent {self.agent_id!r}; output not attached.",
            stacklevel=2,
        )

    def observe_llm_call(
        self,
        prompt: str | None = None,
        response: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> CheckResult:
        """Record a model request/response (for Said, In, InLen, Chars, Tok)."""
        violations: list[EnforcementResult] = []
        with self._lock:
            if prompt:
                violations += self._supervisor.check_action(
                    agent_id=self.agent_id,
                    action="<llm_request>",
                    event_type="llm_request",
                    metadata={"content": prompt, "args": {"char_count": len(prompt)}},
                )
            if response:
                tokens = {
                    k: v
                    for k, v in {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "tokens": (
                            input_tokens + output_tokens
                            if input_tokens is not None and output_tokens is not None
                            else None
                        ),
                    }.items()
                    if v is not None
                }
                violations += self._supervisor.check_action(
                    agent_id=self.agent_id,
                    action="<llm_response>",
                    event_type="llm_response",
                    metadata={"content": response, "args": tokens},
                )
        hard = [r for r in violations if r.action in ("blocked", "escalated")]
        self._record("<llm>", hard)
        return CheckResult(allowed=not any(r.action == "blocked" for r in hard), violations=hard)

    def observe_data_write(self, key: str, fields: list[str] | None = None) -> None:
        """Record that a value containing ``fields`` was written under ``key`` (Has, Flow)."""
        self._supervisor.check_action(
            agent_id=self.agent_id,
            action=f"<data_write:{key}>",
            event_type="data_write",
            metadata={"key": key, "contains": fields},
        )

    def observe_data_read(self, key: str) -> None:
        """Record a read of ``key`` (Flow source)."""
        self._supervisor.check_action(
            agent_id=self.agent_id,
            action=f"<data_read:{key}>",
            event_type="data_read",
            metadata={"key": key},
        )

    def observe_delegation(self, to_agent: str) -> None:
        """Record a hand-off to another agent (Flow sink, Depth)."""
        self._supervisor.check_action(
            agent_id=self.agent_id,
            action=f"<delegate:{to_agent}>",
            event_type="message",
            metadata={"to": to_agent},
        )

    def observe_context(self, facts: dict[str, str]) -> None:
        """Record context facts such as roles or approvals (Ctx, Match)."""
        clean = {k: v for k, v in (facts or {}).items() if k is not None and v is not None}
        if not clean:
            return
        self._supervisor.check_action(
            agent_id=self.agent_id,
            action="<context_update>",
            event_type="context_update",
            metadata={"args": clean},
        )

    def observe_approval(self, role: str, decision: str = "allow", scope: str | None = None) -> None:
        facts = {"approval.role": role, "approval.decision": decision}
        if scope:
            facts["approval.scope"] = scope
        self.observe_context(facts)

    def finish_session(self) -> list[Verdict]:
        """Decide the pending liveness guarantees on the completed trace.

        Unbounded eventualities cannot be refuted mid-session; once the
        session is known to be over, every ``F``/``U`` still pending is a
        violation. Returns the failing verdicts (empty when all obligations
        were discharged). Idempotent until :meth:`reset`.
        """
        from contragent.models.spans import SpanCollector

        with self._lock:
            if self._pending_liveness is not None:
                return list(self._pending_liveness)
            verifier = self._supervisor.verifier
            verifier.set_agents({c.agent.id: c.agent for c in self._system.contracts})
            verifier.sync_from_contracts(self._supervisor.trace, self._system.contracts)
            pending = [
                c
                for c in self._system.contracts
                if c.agent.id == self.agent_id
                and any(getattr(g, "liveness", False) for g in c.guarantees)
            ]
            if not pending:
                self._pending_liveness = []
                return []
            failures: list[Verdict] = []
            with SpanCollector(agent_id=self.agent_id, action="<session_end>") as collector:
                for contract in pending:
                    verdict = verifier.check_contract(contract, include_liveness=True)
                    if not verdict.assumption_holds:
                        continue
                    label = contract.desc or f"{contract.agent.id}: liveness"
                    collector.start_contract_check(label)
                    failed = False
                    for g in verdict.guarantees:
                        if not getattr(g.formula, "liveness", False):
                            continue
                        span = collector.start_guarantee(g.desc)
                        if g.holds:
                            collector.finish_span("ok")
                            continue
                        span.result = False
                        collector.finish_span("violated")
                        details = f"Liveness unmet at session end: {g.desc}"
                        collector.add_violation(kind="liveness", severity="HIGH", evidence=details)
                        collector.add_enforcement(strategy="LivenessEscalate", result_action="escalated")
                        failures.append(g)
                        failed = True
                        self._supervisor._emit(
                            SupervisionEvent(
                                agent_id=self.agent_id,
                                action="<session_end>",
                                constraint_name=f"liveness: {g.desc}",
                                result=EnforcementResult(
                                    action="escalated",
                                    message=f"LIVENESS: {g.desc} - obligation unmet at session end",
                                ),
                            )
                        )
                        self._violations.append(
                            {
                                "tool": "<session_end>",
                                "constraint": f"liveness: {g.desc}",
                                "action": "ESCALATED",
                            }
                        )
                    collector.finish_span("violated" if failed else "ok")
                collector.root.total_contracts_checked = sum(
                    1 for c in collector.root.children if c.span_type == "contragent.contract_check"
                )
                collector.root.violations = len(failures)
                if failures:
                    collector.root.status = "violated"
            self._supervisor._last_turn_span = collector.root
            self._supervisor._turn_spans.append(collector.root)
            self._pending_liveness = failures
            return list(failures)

    # ------------------------------------------------------------------
    # Offline evaluation and analysis
    # ------------------------------------------------------------------

    def evaluate_trace(self, trace: Trace, include_liveness: bool = True) -> dict[str, Any]:
        """Replay a recorded trace and return the end-of-trace verdict.

        Returns a mapping with ``verdict`` (``"FAIL"`` if any contract is
        violated, else ``"PASS"``), ``violations`` (one entry per failing
        guarantee with the contract description), and ``first_violation``
        (the earliest event index at which a guarantee fails, or ``None``).
        """
        from contragent.formulas.evaluator import evaluate as eval_formula
        from contragent.runtime.verifier import TraceVerifier, _raw_formula

        verifier = TraceVerifier()
        verifier.set_agents({c.agent.id: c.agent for c in self._system.contracts})
        verifier.sync_from_contracts(trace, self._system.contracts)
        violations: list[dict] = []
        first: int | None = None
        for contract in self._system.contracts:
            if contract.agent.id != self.agent_id:
                continue
            verdict = verifier.check_contract(contract, include_liveness=include_liveness)
            if not verdict.assumption_holds:
                continue
            for g in verdict.guarantees:
                if g.holds:
                    continue
                violations.append({"contract": contract.desc, "guarantee": g.desc})
                # earliest prefix on which the guarantee already fails
                raw = _raw_formula(g.formula)
                for k in range(1, len(verifier.valuations) + 1):
                    if not eval_formula(raw, verifier.valuations[:k]):
                        first = k - 1 if first is None else min(first, k - 1)
                        break
        return {
            "verdict": "FAIL" if violations else "PASS",
            "violations": violations,
            "first_violation": first,
        }

    def check_conflicts(self, *, backend: str = "auto", **kwargs: Any):
        """Run the library conflict check (see :func:`contragent.analysis.check_conflicts`)."""
        from contragent.analysis import check_conflicts

        return check_conflicts(self._system.contracts, backend=backend, **kwargs)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def supervisor(self) -> Supervisor:
        return self._supervisor

    @property
    def contracts(self) -> list[Contract]:
        return list(self._system.contracts)

    @property
    def trace(self) -> Trace:
        return self._supervisor.trace

    @property
    def violations(self) -> list[dict]:
        return list(self._violations)

    @property
    def check_spans(self) -> list:
        """One span tree per checked action (and one for ``finish_session``)."""
        return self._supervisor.turn_spans

    @property
    def last_check_span(self):
        return self._supervisor.last_turn_span

    def reset(self) -> None:
        with self._lock:
            self._supervisor.reset()
            self._violations.clear()
            self._pending_liveness = None


__all__ = ["ContrAgent", "CheckResult"]
