"""Runtime co-accessibility: can the session still be completed?

The load-time conflict check (:mod:`contragent.analysis.conflicts`) asks
whether a library is satisfiable *at all*, i.e. whether the initial state
is co-accessible.  It says nothing about the states a session actually
reaches.  A sequence of individually legal calls can walk into a state
where every contract is still un-refuted yet no continuation satisfies
them all together -- a *dead end* in the sense of supervisory
control (Ramadge & Wonham 1987); for modular specifications Wonham &
Ramadge (1988) call the failure a *conflict*.

Example.  ``G(called(a) -> X(called(b)))`` together with
``G(Var('count','b') <= 1)`` is satisfiable, so the load-time check
passes.  After one call of ``b``, a call of ``a`` still leaves both
contracts un-refuted, so a per-contract test admits it -- but the next
event must be ``b``, which the bound forbids.

:func:`completable` answers the joint question at runtime: given the
*current residuals* of the loaded contracts, is their conjunction
satisfiable by some continuation?  It is the test a dead-end free mask
performs before exposing a tool.

The residuals come from the monitors
(:meth:`contragent.formulas.dfa_evaluator.DFAEvaluator.residual`), so the
query is over what remains to be satisfied, not over the original
formulas.  Counter bounds are rewritten for the calls already consumed,
because the satisfiability search restarts counters at zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from contragent.formulas.formula import (
    And,
    Const,
    Eq,
    F,
    G,
    Ge,
    Gt,
    Implies,
    Le,
    Lt,
    Not,
    Or,
    U,
    Var,
    X,
)
from contragent.formulas.sat import is_satisfiable

__all__ = ["completable", "rewrite_counter_bounds"]

_COMPARISONS = (Le, Lt, Ge, Gt, Eq)
_BINARY = (And, Or, Implies, U)
_UNARY = (Not, G, F, X)


def _consumed(term: Any, counters: Mapping[str, int]) -> int | None:
    """Calls already made against the counter ``term`` names, if any."""
    if isinstance(term, Var) and term.name == "count":
        args = getattr(term, "args", None) or getattr(term, "key", None)
        tool = args[0] if isinstance(args, (list, tuple)) and args else args
        if isinstance(tool, str):
            return int(counters.get(tool, 0))
    return None


def rewrite_counter_bounds(node: Any, counters: Mapping[str, int]) -> Any:
    """Subtract already-consumed calls from every ``count`` bound in ``node``.

    A residual still carries the library's original bound, e.g.
    ``Var('count','b') <= 1``.  The satisfiability search counts only the
    continuation, so the bound must become ``<= 1 - used(b)``.  A bound
    that drops below zero is unsatisfiable for an upper comparison and is
    folded to ``False``; a lower bound that is already met folds to
    ``True``.
    """
    if isinstance(node, bool) or node is None:
        return node

    if isinstance(node, _COMPARISONS):
        left, right = node.left, node.right
        used = _consumed(left, counters)
        if used is not None and isinstance(right, Const) and isinstance(right.value, (int, float)):
            slack = right.value - used
            if isinstance(node, (Le, Lt)):
                # count_continuation ⋈ slack; a negative slack is unreachable
                if slack < 0 or (isinstance(node, Lt) and slack <= 0):
                    return False
            elif isinstance(node, (Ge, Gt)):
                # already satisfied by the prefix alone
                if slack <= 0 and not isinstance(node, Gt):
                    return True
                if isinstance(node, Gt) and slack < 0:
                    return True
            return type(node)(left, Const(slack))
        return node

    if isinstance(node, _UNARY):
        child = rewrite_counter_bounds(node.child, counters)
        if isinstance(child, bool):
            if isinstance(node, Not):
                return not child
            if isinstance(node, (G, F, X)):
                # G/F/X of a constant is that constant over any trace
                return child
        return type(node)(child)

    if isinstance(node, _BINARY):
        left = rewrite_counter_bounds(node.left, counters)
        right = rewrite_counter_bounds(node.right, counters)
        if isinstance(node, And):
            if left is False or right is False:
                return False
            if left is True:
                return right
            if right is True:
                return left
        elif isinstance(node, Or):
            if left is True or right is True:
                return True
            if left is False:
                return right
            if right is False:
                return left
        elif isinstance(node, Implies):
            if left is False or right is True:
                return True
            if left is True:
                return right
        return type(node)(left, right)

    return node


def completable(
    residuals: Iterable[Any],
    *,
    counters: Mapping[str, int] | None = None,
    max_states: int = 50_000,
    max_alphabet: int = 4096,
    theory_checker: Any = "auto",
    exact_counters: bool = True,
) -> bool | None:
    """Is the conjunction of ``residuals`` satisfiable by some continuation?

    Args:
        residuals: The monitors' current residuals.  ``True`` marks a
            contract already settled satisfied, ``False`` one already
            refuted.
        counters: Calls consumed so far per tool, used to rewrite the
            ``count`` bounds the residuals still carry.
        max_states / max_alphabet / theory_checker / exact_counters:
            Passed through to
            :func:`contragent.formulas.sat.is_satisfiable`.

    Returns:
        ``True`` when a continuation satisfies every residual, ``False``
        when none does, and ``None`` when the search exceeds its budget.
        A ``None`` must be treated as ``True`` by a mask that may not
        hide a legal tool.
    """
    counters = counters or {}
    pending: list[Any] = []
    for residual in residuals:
        if residual is False:
            return False
        if residual is True or residual is None:
            continue
        rewritten = rewrite_counter_bounds(residual, counters)
        if rewritten is False:
            return False
        if rewritten is True:
            continue
        pending.append(rewritten)

    if not pending:
        return True

    from contragent.analysis.conflicts import derive_domain_constraints

    mutex_groups, implications, axioms = derive_domain_constraints(pending)
    return is_satisfiable(
        pending + list(axioms),
        mutex_groups=mutex_groups,
        implications=implications,
        max_states=max_states,
        max_alphabet=max_alphabet,
        theory_checker=theory_checker,
        exact_counters=exact_counters,
    )
