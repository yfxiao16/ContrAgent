"""Static check: can the one-step mask walk this library into a dead end?

The load-time conflict check asks whether a library is satisfiable at
all.  It does not rule out a *dead end*: a state every contract still
permits, from which no continuation satisfies them together.  The
runtime remedy is a co-accessibility query
(:mod:`contragent.analysis.completion`), one satisfiability call per
control state.  This module is the cheaper first tier -- a syntactic
check on the library that, when it passes, means the one-step mask
cannot block and the query is never needed.

Two patterns give a library a dead end, and both are decidable from the
formulas alone:

``rate-limited obligation``
    An obligation ``G(trig -> X t)`` or ``G(trig -> F t)`` whose
    obligated tool ``t`` also carries a bound ``count(t) <= N``.  Once
    the bound is spent, firing the trigger leaves the obligation
    undischargeable.

``competing next-obligations``
    Two ``X`` obligations sharing a trigger but naming different tools.
    ``X`` admits one next event, so at most one can be discharged.  This
    pattern has no ``F`` analogue: two ``eventually`` obligations can be
    discharged in turn.

A library that exhibits neither is not thereby proved dead-end free --
two further conditions of the accompanying proposition (the obligated
tool's requirements implied by the trigger, and no prohibition on it
conditioned on a fact the trigger implies) need semantic reasoning and
are reported as unchecked.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from contragent.formulas.formula import (
    And,
    Atom,
    Const,
    F,
    G,
    Implies,
    Le,
    Lt,
    Var,
    X,
)

__all__ = ["Obligation", "DeadEndReport", "check_dead_ends"]


@dataclass(frozen=True)
class Obligation:
    """A guarantee of the form ``G(trigger -> X tool)`` or ``... F tool``."""

    label: str
    trigger: str
    tool: str
    operator: str  # "X" or "F"


@dataclass
class DeadEndReport:
    """What the syntactic check found."""

    obligations: list[Obligation] = field(default_factory=list)
    bounded_tools: dict[str, int] = field(default_factory=dict)
    rate_limited: list[Obligation] = field(default_factory=list)
    competing: list[tuple[Obligation, Obligation]] = field(default_factory=list)
    unchecked: tuple[str, ...] = (
        "the obligated tool's requirements are implied by the trigger",
        "no prohibition on the obligated tool is conditioned on a fact the trigger implies",
        "no return retracts a fact",
    )

    @property
    def ok(self) -> bool:
        """True when neither decidable dead-end pattern occurs."""
        return not self.rate_limited and not self.competing

    def summary(self) -> str:
        if self.ok:
            return (
                f"no dead-end pattern in {len(self.obligations)} obligation(s); "
                f"{len(self.unchecked)} condition(s) left unchecked"
            )
        parts = []
        if self.rate_limited:
            parts.append(f"{len(self.rate_limited)} rate-limited obligation(s)")
        if self.competing:
            parts.append(f"{len(self.competing)} competing next-obligation pair(s)")
        return "; ".join(parts)


def _called_tool(node: Any) -> str | None:
    """The tool name of a ``called(t)`` atom, if that is what ``node`` is."""
    if isinstance(node, Atom):
        key = node.key() if callable(getattr(node, "key", None)) else None
        if isinstance(key, str) and key.startswith("called(") and key.endswith(")"):
            return key[len("called(") : -1]
    return None


def _conjuncts(node: Any) -> list[Any]:
    if isinstance(node, And):
        return _conjuncts(node.left) + _conjuncts(node.right)
    return [node]


def _obligations_in(node: Any, label: str, out: list[Obligation]) -> None:
    """Collect ``G(trig -> X/F tool)`` shapes anywhere in ``node``."""
    if isinstance(node, G):
        body = node.child
        if isinstance(body, Implies):
            trigger = _called_tool(body.left)
            head = body.right
            if trigger and isinstance(head, (X, F)):
                tool = _called_tool(head.child)
                if tool:
                    out.append(
                        Obligation(label, trigger, tool, "X" if isinstance(head, X) else "F")
                    )
                    return
        _obligations_in(body, label, out)
        return
    for attr in ("left", "right", "child"):
        child = getattr(node, attr, None)
        if child is not None and not isinstance(child, (str, int, float, bool)):
            _obligations_in(child, label, out)


def _bounds_in(node: Any, out: dict[str, int]) -> None:
    """Collect ``count(t) <= N`` / ``< N`` upper bounds."""
    if isinstance(node, (Le, Lt)):
        left, right = node.left, node.right
        if (
            isinstance(left, Var)
            and left.name == "count"
            and left.args
            and isinstance(right, Const)
            and isinstance(right.value, (int, float))
        ):
            tool = left.args[0]
            bound = int(right.value) - (1 if isinstance(node, Lt) else 0)
            out[tool] = min(out.get(tool, bound), bound)
        return
    for attr in ("left", "right", "child"):
        child = getattr(node, attr, None)
        if child is not None and not isinstance(child, (str, int, float, bool)):
            _bounds_in(child, out)


def _normalise(contracts: Iterable[Any]) -> list[tuple[str, Any]]:
    """Pull ``(label, guarantee formula)`` out of either Contract flavour.

    ContrAgent contracts carry ``guarantee``/``desc``; the agent-side
    contracts of the control layer carry ``formula``/``name``.  Taking
    both keeps the check usable from either side without importing one
    into the other.
    """
    out: list[tuple[str, Any]] = []
    for index, contract in enumerate(contracts):
        formula = getattr(contract, "guarantee", None)
        if formula is None:
            formula = getattr(contract, "formula", None)
        if formula is None:
            continue
        label = (
            getattr(contract, "desc", None)
            or getattr(contract, "name", None)
            or f"contract[{index}]"
        )
        for part in formula if isinstance(formula, (list, tuple)) else [formula]:
            out.append((str(label), part))
    return out


def check_dead_ends(contracts: Sequence[Any] | Iterable[Any]) -> DeadEndReport:
    """Run the syntactic dead-end free check over a contract library.

    Args:
        contracts: The loaded library, as
            :func:`contragent.analysis.conflicts.check_conflicts` takes it.

    Returns:
        A :class:`DeadEndReport`.  ``report.ok`` means neither decidable
        dead-end pattern occurs, so the one-step mask needs no
        co-accessibility query on this library, subject to
        ``report.unchecked``.
    """
    report = DeadEndReport()

    for label, formula in _normalise(contracts):
        for conjunct in _conjuncts(formula):
            _obligations_in(conjunct, label, report.obligations)
            _bounds_in(conjunct, report.bounded_tools)

    for obligation in report.obligations:
        if obligation.tool in report.bounded_tools:
            report.rate_limited.append(obligation)

    next_obligations = [o for o in report.obligations if o.operator == "X"]
    for i, first in enumerate(next_obligations):
        for second in next_obligations[i + 1 :]:
            if first.trigger == second.trigger and first.tool != second.tool:
                report.competing.append((first, second))

    return report
