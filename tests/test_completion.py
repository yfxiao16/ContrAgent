"""Runtime completion check: dead ends the load-time check misses."""

from contragent.analysis.completion import completable, rewrite_counter_bounds
from contragent.formulas.dfa_evaluator import DFAEvaluator
from contragent.formulas.formula import Const, Le, Var
from contragent.formulas.parser import parse_repr


def _call(tool):
    return {f"called({tool})": True}


def _residuals(formulas, events):
    """Step one evaluator per formula through ``events``; return the residuals."""
    evaluators = [DFAEvaluator(f) for f in formulas]
    for event in events:
        for e in evaluators:
            e.step(event)
    return [e.residual for e in evaluators]


def test_rewrite_subtracts_consumed_calls():
    node = Le(Var("count", "b"), Const(3))
    assert rewrite_counter_bounds(node, {"b": 1}) == Le(Var("count", "b"), Const(2))


def test_rewrite_folds_an_exhausted_bound_to_false():
    node = Le(Var("count", "b"), Const(1))
    assert rewrite_counter_bounds(node, {"b": 2}) is False


def test_obligation_plus_rate_limit_blocks_after_the_trigger():
    """The appendix's counterexample, at the state the mask would probe.

    ``G(a -> X b)`` with ``count(b) <= 1``: both are satisfiable together,
    so the load-time check passes.  After one call of ``b`` and then a
    call of ``a`` the next event must be ``b``, which the bound forbids.
    """
    formulas = [
        parse_repr("G(called('a') -> X(called('b')))"),
        parse_repr("G((Var('count','b') <= 1))"),
    ]
    # b consumed once, then the candidate call of a
    residuals = _residuals(formulas, [_call("b"), _call("a")])
    assert completable(residuals, counters={"b": 1}) is False


def test_same_state_is_fine_before_the_trigger():
    """Without the call of ``a`` the state is still completable."""
    formulas = [
        parse_repr("G(called('a') -> X(called('b')))"),
        parse_repr("G((Var('count','b') <= 1))"),
    ]
    residuals = _residuals(formulas, [_call("b")])
    assert completable(residuals, counters={"b": 1}) is not False


def test_two_next_obligations_on_one_trigger_block():
    """Prescriptive guarantees is a dead end without any liveness guarantee.

    ``X`` admits one next event, so two obligations on the same trigger
    cannot both be discharged.
    """
    formulas = [
        parse_repr("G(called('a') -> X(called('b')))"),
        parse_repr("G(called('a') -> X(called('c')))"),
    ]
    residuals = _residuals(formulas, [_call("a")])
    assert completable(residuals, counters={}) is False


def test_settled_refutation_short_circuits():
    assert completable([False, True]) is False


def test_all_settled_satisfied_is_completable():
    assert completable([True, True]) is True
