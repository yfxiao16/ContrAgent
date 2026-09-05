"""Tests for the native LTLf satisfiability engine (formulas/sat.py)."""

from contragent.formulas.formula import (
    And,
    Atom,
    Const,
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


def _called(tool: str) -> Atom:
    return Atom("called", tool)


A = _called("a")
B = _called("b")


# ---------------------------------------------------------------------------
# Propositional / basic
# ---------------------------------------------------------------------------


class TestBasics:
    def test_single_atom_sat(self):
        assert is_satisfiable(A) is True

    def test_contradiction_unsat(self):
        assert is_satisfiable(And(A, Not(A))) is False

    def test_or_with_contradiction_sat(self):
        assert is_satisfiable(Or(And(A, Not(A)), B)) is True

    def test_implies_sat(self):
        assert is_satisfiable(Implies(A, B)) is True

    def test_empty_conjunction_sat(self):
        assert is_satisfiable([]) is True

    def test_list_is_conjoined(self):
        assert is_satisfiable([A, Not(A)]) is False
        assert is_satisfiable([A, B]) is True


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------


class TestTemporal:
    def test_eventually_sat(self):
        assert is_satisfiable(F(A)) is True

    def test_globally_sat(self):
        assert is_satisfiable(G(A)) is True

    def test_safety_vs_liveness_unsat(self):
        assert is_satisfiable(And(G(Not(A)), F(A))) is False

    def test_next_sat(self):
        assert is_satisfiable(X(A)) is True

    def test_until_sat(self):
        assert is_satisfiable(U(Not(B), A)) is True

    def test_until_blocked_unsat(self):
        # b must stay off until a fires, but a never may fire and b must.
        assert is_satisfiable([U(Not(B), A), G(Not(A)), F(B)]) is False

    def test_response_sat(self):
        assert is_satisfiable([G(Implies(A, F(B))), F(A)]) is True

    def test_ping_pong_unsat(self):
        # The mus2muc README example: the last a/b event always leaves
        # an undischargeable X-obligation, so no finite trace satisfies
        # all four (matches the runtime monitor's verdict).
        formulas = [F(A), F(B), G(Implies(A, X(B))), G(Implies(B, X(A)))]
        assert is_satisfiable(formulas) is False

    def test_trailing_next_is_weak(self):
        # An un-consumed X at trace end collapses to ⊤ (weak next),
        # mirroring dfa_evaluator._finalize: X(X(a)) is satisfied by a
        # single-event trace because the inner X is never even peeled.
        assert is_satisfiable(And(X(X(A)), G(Not(A)))) is True

    def test_negated_temporal(self):
        assert is_satisfiable(Not(G(A))) is True
        assert is_satisfiable(And(Not(F(A)), F(A))) is False


# ---------------------------------------------------------------------------
# Arithmetic abstraction
# ---------------------------------------------------------------------------


class TestArithmeticAbstraction:
    def test_complement_pair_unsat(self):
        x = Var("count", "x")
        # Le and Gt over identical terms share one abstract proposition.
        assert is_satisfiable(And(Le(x, Const(5)), Gt(x, Const(5)))) is False

    def test_constant_ordering_conflict_detected(self):
        x = Var("count", "x")
        # The pointwise theory checker relates different constants over
        # one register: <=5 and >=10 cannot hold at the same event.
        assert is_satisfiable(And(Le(x, Const(5)), Ge(x, Const(10)))) is False

    def test_abstraction_escape_hatch(self):
        x = Var("count", "x")
        # With both the theory checker and the counter gadget disabled,
        # the pure boolean abstraction treats the two comparisons as
        # independent propositions.
        assert (
            is_satisfiable(
                And(Le(x, Const(5)), Ge(x, Const(10))),
                theory_checker=None,
                exact_counters=False,
            )
            is True
        )

    def test_temporal_constant_ordering_unsat(self):
        x = Var("count", "x")
        # F forces an event where count >= 10 while G bounds it by 3
        # at every event — pointwise theory catches the clash.
        assert is_satisfiable([G(Le(x, Const(3))), F(Ge(x, Const(10)))]) is False

    def test_compatible_bounds_sat(self):
        x = Var("count", "x")
        assert is_satisfiable([G(Le(x, Const(10))), F(Ge(x, Const(3)))]) is True

    def test_counter_integrality(self):
        x = Var("count", "x")
        # 0 < count < 1 has no integer solution.
        assert is_satisfiable(And(Gt(x, Const(0)), Lt(x, Const(1)))) is False

    def test_counter_non_negative(self):
        x = Var("count", "x")
        assert is_satisfiable(Lt(x, Const(0))) is False

    def test_temporal_over_comparison(self):
        x = Var("count", "x")
        assert is_satisfiable([G(Le(x, Const(3))), F(Gt(x, Const(3)))]) is False

    def test_partial_terms_not_complement_paired(self):
        from contragent.formulas.formula import ArgValue

        a = ArgValue("transfer", "amount")
        # At an event where the arg is missing, the runtime evaluates
        # BOTH Le and Gt to False — so their negations can hold
        # together and this must NOT be reported as a conflict.
        both_negated = And(G(Not(Le(a, Const(5)))), G(Not(Gt(a, Const(5)))))
        assert is_satisfiable(both_negated) is True

    def test_partial_terms_positive_mutex(self):
        from contragent.formulas.formula import ArgValue

        a = ArgValue("transfer", "amount")
        # Both *holding* at one event is still impossible — the theory
        # checker restores the mutual exclusion soundly.
        assert is_satisfiable(And(Le(a, Const(5)), Gt(a, Const(5)))) is False


# ---------------------------------------------------------------------------
# Saturating-counter gadget (exact counter semantics)
# ---------------------------------------------------------------------------


def _at_least_calls(atom: Atom, n: int):
    """``atom`` holds on at least n distinct events (pure temporal form)."""
    f = F(atom)
    for _ in range(n - 1):
        f = F(And(atom, X(f)))
    return f


class TestCounterGadget:
    X_VAR = Var("count", "x")
    CALLED_X = _called("x")

    def test_cross_vocabulary_conflict_caught(self):
        # rate_limit-style bound on the counter vs. four *call events*:
        # only the increment-by-one semantics links the two vocabularies.
        formulas = [G(Le(self.X_VAR, Const(3))), _at_least_calls(self.CALLED_X, 4)]
        assert is_satisfiable(formulas) is False
        # Escape hatch: without the gadget the clash is invisible.
        assert is_satisfiable(formulas, exact_counters=False) is True

    def test_boundary_is_exact(self):
        # Exactly 3 calls fit under a cap of 3 — the gadget must not
        # over-tighten.
        assert (
            is_satisfiable(
                [G(Le(self.X_VAR, Const(3))), _at_least_calls(self.CALLED_X, 3)]
            )
            is True
        )

    def test_counter_cannot_grow_without_calls(self):
        formulas = [F(Ge(self.X_VAR, Const(4))), G(Not(self.CALLED_X))]
        assert is_satisfiable(formulas) is False
        assert is_satisfiable(F(Ge(self.X_VAR, Const(4)))) is True

    def test_monotonicity_is_built_in(self):
        # "reach 2, later drop to <= 1" — impossible for a monotone
        # counter; the gadget needs no separate persistence axiom.
        formula = F(And(Ge(self.X_VAR, Const(2)), F(Le(self.X_VAR, Const(1)))))
        assert is_satisfiable(formula) is False

    def test_equality_over_counter(self):
        from contragent.formulas.formula import Eq

        assert is_satisfiable(F(Eq(self.X_VAR, Const(2)))) is True
        # count == 2 can never hold after count >= 3 (monotone).
        assert (
            is_satisfiable(
                F(And(Ge(self.X_VAR, Const(3)), F(Eq(self.X_VAR, Const(2)))))
            )
            is False
        )

    def test_count_with_register(self):
        cw = Atom("called_with", "x", "rm")
        v = Var("count_with", "x", "rm")
        assert is_satisfiable([F(Ge(v, Const(2))), G(Not(cw))]) is False
        assert is_satisfiable(F(Ge(v, Const(2)))) is True

    def test_cap_fallback_is_sound(self):
        # A threshold above the gadget cap falls back to the free-prop
        # treatment: the clash goes undetected (spurious SAT) but the
        # engine must not crash or report a false conflict.
        formulas = [
            G(Le(self.X_VAR, Const(10**6))),
            _at_least_calls(self.CALLED_X, 2),
        ]
        assert is_satisfiable(formulas) is True


# ---------------------------------------------------------------------------
# Domain constraints
# ---------------------------------------------------------------------------


class TestDomainConstraints:
    def test_mutex_makes_unsat(self):
        mutex = [["called(a)", "called(b)"]]
        formulas = [G(A), F(B)]
        assert is_satisfiable(formulas) is True
        assert is_satisfiable(formulas, mutex_groups=mutex) is False

    def test_mutex_still_allows_interleaving(self):
        mutex = [["called(a)", "called(b)"]]
        assert is_satisfiable([F(A), F(B)], mutex_groups=mutex) is True

    def test_implication_makes_unsat(self):
        cw = Atom("called_with", "x", "rm -rf")
        cx = _called("x")
        impl = [(cw.key(), cx.key())]
        formulas = [F(cw), G(Not(cx))]
        assert is_satisfiable(formulas) is True
        assert is_satisfiable(formulas, implications=impl) is False


# ---------------------------------------------------------------------------
# Budgets and wrappers
# ---------------------------------------------------------------------------


class TestBudgetsAndWrappers:
    def test_alphabet_budget_unknown(self):
        formulas = [F(_called(f"t{i}")) for i in range(8)]
        assert is_satisfiable(formulas, max_alphabet=4) is None

    def test_state_budget_unknown_or_decided(self):
        # A tiny state cap must never produce a wrong verdict — only
        # True (early accept) or None.
        verdict = is_satisfiable([G(Not(A)), F(A)], max_states=1)
        assert verdict in (False, None)

    def test_detformula_unwrapped(self):
        from tests._builders import must_precede, rate_limit

        assert is_satisfiable(rate_limit("x", 2)) is True
        assert is_satisfiable(must_precede("approve", "pay")) is True

    def test_pattern_conflict(self):
        from tests._builders import must_precede

        # "approve before pay" + "approve never called" + "pay happens"
        assert (
            is_satisfiable(
                [
                    must_precede("approve", "pay"),
                    G(Not(_called("approve"))),
                    F(_called("pay")),
                ]
            )
            is False
        )
