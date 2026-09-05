"""Tests for pointwise theory checking (formulas/theory.py)."""

import pytest

from contragent.formulas.formula import ArgValue, Const, CtxValue, Var
from contragent.formulas.theory import (
    IntervalChecker,
    TheoryLiteral,
    default_theory_checker,
)

X = Var("count", "x")
AMOUNT = ArgValue("transfer", "amount")
APPROVED = CtxValue("approved_amount")


def lit(op, left, right, *, total=True, positive=True):
    if isinstance(right, (int, float)):
        right = Const(right)
    return TheoryLiteral(op=op, left=left, right=right, total=total, positive=positive)


class TestIntervalChecker:
    def setup_method(self):
        self.chk = IntervalChecker()

    def test_empty_consistent(self):
        assert self.chk.consistent([]) is True

    def test_disjoint_bounds_inconsistent(self):
        assert (
            self.chk.consistent([lit("le", X, 5), lit("lt", X, 10, positive=False)])
            is False
        )
        # x <= 5 and x >= 10  (Ge arrives as negated lt)

    def test_overlapping_bounds_consistent(self):
        assert (
            self.chk.consistent([lit("le", X, 10), lit("lt", X, 3, positive=False)])
            is True
        )

    def test_negated_le_is_strict_lower(self):
        # !(x <= 5) means x > 5; with x <= 6 still consistent (x = 6).
        assert (
            self.chk.consistent([lit("le", X, 5, positive=False), lit("le", X, 6)])
            is True
        )
        assert (
            self.chk.consistent([lit("le", X, 5, positive=False), lit("le", X, 5)])
            is False
        )

    def test_counter_integrality(self):
        # 0 < count < 1 has no integer solution.
        assert (
            self.chk.consistent([lit("le", X, 0, positive=False), lit("lt", X, 1)])
            is False
        )

    def test_counter_non_negative(self):
        assert self.chk.consistent([lit("lt", X, 0)]) is False

    def test_eq_point_solution(self):
        assert self.chk.consistent([lit("eq", X, 3), lit("le", X, 3)]) is True
        assert self.chk.consistent([lit("eq", X, 3), lit("lt", X, 3)]) is False

    def test_eq_excluded_point(self):
        # x <= 3 and x >= 3 and x != 3 — inconsistent.
        lits = [
            lit("le", X, 3),
            lit("lt", X, 3, positive=False),
            lit("eq", X, 3, positive=False),
        ]
        assert self.chk.consistent(lits) is False

    def test_fractional_eq_on_counter(self):
        assert self.chk.consistent([lit("eq", X, 2.5)]) is False

    def test_partial_negative_is_ignored(self):
        # !(amount <= 5) may just mean "amount missing" — no constraint,
        # so combining with !(amount > 5) stays consistent.
        lits = [
            lit("le", AMOUNT, 5, total=False, positive=False),
            lit("gt", AMOUNT, 5, total=False, positive=False),
        ]
        assert self.chk.consistent(lits) is True

    def test_partial_positive_pair_inconsistent(self):
        lits = [
            lit("le", AMOUNT, 5, total=False),
            lit("gt", AMOUNT, 5, total=False),
        ]
        assert self.chk.consistent(lits) is False

    def test_term_vs_term_skipped(self):
        # Interval checker cannot relate two runtime terms — must stay
        # conservative (consistent), never crash.
        literal = TheoryLiteral(
            op="le", left=AMOUNT, right=APPROVED, total=False, positive=True
        )
        assert self.chk.consistent([literal]) is True


class TestZ3Checker:
    def setup_method(self):
        z3 = pytest.importorskip("z3")
        del z3
        from contragent.formulas.theory import Z3Checker

        self.chk = Z3Checker()

    def test_basic_bounds(self):
        assert (
            self.chk.consistent([lit("le", X, 5), lit("lt", X, 10, positive=False)])
            is False
        )
        assert (
            self.chk.consistent([lit("le", X, 10), lit("lt", X, 3, positive=False)])
            is True
        )

    def test_counter_integrality_and_sign(self):
        assert (
            self.chk.consistent([lit("le", X, 0, positive=False), lit("lt", X, 1)])
            is False
        )
        assert self.chk.consistent([lit("lt", X, 0)]) is False

    def test_term_vs_term_transitivity(self):
        # amount <= approved, approved <= 100, amount > 100 → inconsistent.
        lits = [
            TheoryLiteral(
                op="le", left=AMOUNT, right=APPROVED, total=False, positive=True
            ),
            lit("le", APPROVED, 100, total=False),
            lit("gt", AMOUNT, 100, total=False),
        ]
        assert self.chk.consistent(lits) is False

    def test_partial_negative_no_constraint(self):
        lits = [
            lit("le", AMOUNT, 5, total=False, positive=False),
            lit("gt", AMOUNT, 5, total=False, positive=False),
        ]
        assert self.chk.consistent(lits) is True


class TestDefaultChecker:
    def test_returns_a_checker(self):
        chk = default_theory_checker()
        assert chk.consistent([lit("le", X, 5)]) is True
        assert default_theory_checker() is chk  # cached singleton
