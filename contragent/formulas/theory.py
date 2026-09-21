"""Pointwise theory checking for the LTLf satisfiability abstraction.

:mod:`contragent.formulas.sat` abstracts arithmetic comparisons into
opaque propositions, which loses their numeric meaning: nothing stops a
propositional model from asserting ``count <= 3`` and ``count >= 10``
at the same timestep. This module restores the *pointwise* theory: a
checker decides whether a conjunction of comparison literals — one
timestep's assignment to the comparison propositions — is consistent
with linear arithmetic. The satisfiability search drops every abstract
valuation whose comparison projection is theory-inconsistent, which is
sound (no real trace can ground an inconsistent conjunction) and lifts
the "``x <= 3`` and ``x >= 10`` are independent" blind spot.

This is the lazy-SMT recipe: propositional/temporal reasoning stays in
the automaton search, theory reasoning happens per candidate valuation.
Cross-timestep arithmetic (counter monotonicity) is handled separately
by persistence axioms in :mod:`contragent.analysis.conflicts`.

Missing-value semantics
-----------------------
The runtime evaluates a comparison to ``False`` when either operand is
missing (``Term.evaluate`` returned ``None``). Comparisons over *total*
terms (``Var``/``Const``, which default to 0) behave classically: the
literal's polarity always constrains the value. Comparisons over
*partial* terms (``ArgValue``, ``CtxValue``, ...) only constrain the
theory when **positive** — a negative literal may just mean "the value
was absent", so it is ignored. Checkers must honour the
``TheoryLiteral.total`` flag accordingly.

Backends
--------
* :class:`IntervalChecker` — pure Python, zero dependencies. Reasons
  per term about ``term <op> Const`` bounds (with integrality for
  ``count``-style registers). Ignores what it cannot see — always
  conservative.
* :class:`Z3Checker` — full linear arithmetic via ``z3-solver``
  (``pip install contragent[smt]``). Also handles term-vs-term
  comparisons (e.g. ``args.amount <= ctx.approved_amount``) and
  transitivity across terms.

:func:`default_theory_checker` returns Z3 when importable, else the
interval checker. Every checker is *conservative*: on anything it
cannot decide it answers "consistent", so enabling theory checking can
only turn spurious-SAT verdicts into sound UNSAT ones, never the
reverse.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from contragent.formulas.formula import Const, Var

# Var names whose registers are integer-valued (grounding counts events).
_INTEGER_VAR_NAMES = frozenset(
    {"count", "count_with", "consecutive_count", "token_count", "delegation_depth"}
)

# Var names whose registers can never go below zero.
_NON_NEGATIVE_VAR_NAMES = _INTEGER_VAR_NAMES


@dataclass(frozen=True)
class TheoryLiteral:
    """One comparison proposition with its assigned polarity.

    Attributes:
        op: ``"le"`` | ``"lt"`` | ``"ge"`` | ``"gt"`` | ``"eq"``. For
            total comparisons the sat-module leaf table canonicalizes
            ``Gt``/``Ge`` into negated ``le``/``lt``, so only
            ``le``/``lt``/``eq`` appear; partial comparisons keep their
            original orientation (their negation is not the complement
            comparison — see ``total``).
        left / right: The original ``Term`` operands.
        total: True when both operands always produce a value (Var /
            Const). When False, a negative literal carries no theory
            content (the runtime may have evaluated it False because a
            value was missing).
        positive: The polarity assigned by the candidate valuation.
    """

    op: str
    left: Any
    right: Any
    total: bool
    positive: bool


class TheoryChecker(Protocol):
    """Decides pointwise consistency of comparison-literal conjunctions."""

    def consistent(self, literals: Sequence[TheoryLiteral]) -> bool:
        """Return False only if the conjunction is certainly unsatisfiable."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Pure-Python interval backend
# ---------------------------------------------------------------------------


def _is_integer_term(term: Any) -> bool:
    return isinstance(term, Var) and term.name in _INTEGER_VAR_NAMES


def _is_non_negative_term(term: Any) -> bool:
    return isinstance(term, Var) and term.name in _NON_NEGATIVE_VAR_NAMES


class IntervalChecker:
    """Per-term bound propagation for ``term <op> Const`` literals.

    For each distinct left-hand term, intersects the intervals implied
    by its literals (integer-tightened for count-style registers, with
    an implicit ``>= 0`` for counters). Literals it cannot interpret —
    term-vs-term comparisons, non-numeric constants — are skipped, so
    the verdict stays conservative.
    """

    def consistent(self, literals: Sequence[TheoryLiteral]) -> bool:
        by_term: dict[Any, list[TheoryLiteral]] = {}
        for lit in literals:
            if not isinstance(lit.right, Const):
                continue
            if not isinstance(lit.right.value, (int, float)):
                continue
            if not lit.total and not lit.positive:
                continue  # negative over a partial term: no information
            by_term.setdefault(lit.left, []).append(lit)
        return all(self._group_consistent(term, lits) for term, lits in by_term.items())

    @staticmethod
    def _group_consistent(term: Any, lits: list[TheoryLiteral]) -> bool:
        integral = _is_integer_term(term)
        lo, hi = -math.inf, math.inf  # inclusive bounds after tightening
        excluded: set[float] = set()
        if _is_non_negative_term(term):
            lo = 0

        def tighten_upper(c: float, strict: bool) -> float:
            if not strict:
                return c
            # x < c  →  x <= c - 1 for integers, else open bound.
            return c - 1 if integral else math.nextafter(c, -math.inf)

        def tighten_lower(c: float, strict: bool) -> float:
            if not strict:
                return c
            return c + 1 if integral else math.nextafter(c, math.inf)

        for lit in lits:
            c = float(lit.right.value)
            # Normalize each literal to "x <= c" / "x >= c" (strict or
            # not). Polarity flips the operator; partial negatives were
            # filtered out by the caller.
            if lit.op == "le":
                op = "le" if lit.positive else "gt"
            elif lit.op == "lt":
                op = "lt" if lit.positive else "ge"
            elif lit.op == "ge":
                op = "ge" if lit.positive else "lt"
            elif lit.op == "gt":
                op = "gt" if lit.positive else "le"
            elif lit.op == "eq":
                if lit.positive:
                    if integral and c != math.floor(c):
                        return False  # count == 2.5 is impossible
                    lo, hi = max(lo, c), min(hi, c)
                else:
                    excluded.add(c)
                continue
            else:  # pragma: no cover - unknown op, stay conservative
                continue
            if integral:
                # Fractional constants collapse onto the integer grid:
                # x <= 2.5 ≡ x <= 2, x >= 2.5 ≡ x >= 3, etc.
                c = math.floor(c) if op in ("le", "gt") else math.ceil(c)
            if op == "le":
                hi = min(hi, c)
            elif op == "lt":
                hi = min(hi, tighten_upper(c, strict=True))
            elif op == "ge":
                lo = max(lo, c)
            elif op == "gt":
                lo = max(lo, tighten_lower(c, strict=True))
        if lo > hi:
            return False
        if lo == hi and lo in excluded:
            return False
        return True


# ---------------------------------------------------------------------------
# Z3 backend (optional dependency)
# ---------------------------------------------------------------------------


class Z3Checker:
    """Linear-arithmetic consistency via z3 (``pip install contragent[smt]``).

    Each distinct term becomes one z3 variable (``Int`` for count-style
    registers, ``Real`` otherwise); each partial term additionally gets
    a presence boolean so that a positive literal over it asserts
    "present and satisfied" while a negative literal asserts nothing.
    Unlike :class:`IntervalChecker` this also relates *different* terms
    (``args.amount <= ctx.approved_amount`` plus transitive chains).
    """

    def __init__(self) -> None:
        import z3  # deferred so the module imports without z3 installed

        self._z3 = z3
        self._cache: dict[frozenset, bool] = {}

    def consistent(self, literals: Sequence[TheoryLiteral]) -> bool:
        key = frozenset(literals)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        verdict = self._check(literals)
        self._cache[key] = verdict
        return verdict

    def _check(self, literals: Sequence[TheoryLiteral]) -> bool:
        z3 = self._z3
        solver = z3.Solver()
        terms: dict[Any, Any] = {}
        present: dict[Any, Any] = {}

        def var_of(term: Any) -> Any:
            if term in terms:
                return terms[term]
            n = len(terms)
            v = z3.Int(f"t{n}") if _is_integer_term(term) else z3.Real(f"t{n}")
            if _is_non_negative_term(term):
                solver.add(v >= 0)
            terms[term] = v
            return v

        def presence_of(term: Any) -> Any:
            if term not in present:
                present[term] = z3.Bool(f"present{len(present)}")
            return present[term]

        for lit in literals:
            operands = []
            ok = True
            for side in (lit.left, lit.right):
                if isinstance(side, Const):
                    if not isinstance(side.value, (int, float)):
                        ok = False
                        break
                    operands.append(side.value)
                else:
                    operands.append(var_of(side))
            if not ok:
                continue  # non-numeric constant — skip conservatively
            left, right = operands
            if lit.op == "le":
                cmp = left <= right
            elif lit.op == "lt":
                cmp = left < right
            elif lit.op == "ge":
                cmp = left >= right
            elif lit.op == "gt":
                cmp = left > right
            else:
                cmp = left == right
            if lit.total:
                solver.add(cmp if lit.positive else z3.Not(cmp))
            else:
                guards = [presence_of(s) for s in (lit.left, lit.right) if not isinstance(s, Const)]
                holds = z3.And(*guards, cmp)
                if lit.positive:
                    solver.add(holds)
                # A negative literal over a partial term is vacuously
                # explainable by absence — no constraint.
        result = solver.check()
        # ``unknown`` must stay conservative.
        return result != self._z3.unsat


# ---------------------------------------------------------------------------
# Default resolution
# ---------------------------------------------------------------------------

_DEFAULT: TheoryChecker | None = None
_DEFAULT_RESOLVED = False


def default_theory_checker() -> TheoryChecker:
    """Z3 when installed, otherwise the pure-Python interval checker."""
    global _DEFAULT, _DEFAULT_RESOLVED
    if not _DEFAULT_RESOLVED:
        try:
            _DEFAULT = Z3Checker()
        except ImportError:
            _DEFAULT = IntervalChecker()
        _DEFAULT_RESOLVED = True
    assert _DEFAULT is not None
    return _DEFAULT
