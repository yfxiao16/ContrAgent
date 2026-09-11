"""Tests for the contract-library checks (analysis/conflicts.py).

Covers the paper's library check: joint satisfiability of ⋀ᵢ(Aᵢ ∧ Gᵢ)
and minimal-unsatisfiable-core extraction to name the contracts that
cannot hold together.
"""

import pytest

from contragent.analysis import check_conflicts, extract_muc
from contragent.analysis.conflicts import _units_of, derive_domain_constraints
from contragent.analysis.mus2muc_backend import (
    Mus2mucUnavailable,
    is_available,
    unavailable_reason,
    units_to_ltlfconj,
)
from contragent.formulas.formula import And, Atom, F, G, Not
from contragent.models.agent import Agent
from contragent.models.contract import Contract


def _called(tool: str) -> Atom:
    return Atom("called", tool)


def _leaks(tool: str) -> Atom:
    """An environment predicate: ``tool`` returned a secret."""
    return Atom("output_has", tool, "SECRET")


AGENT = Agent(id="bot")


def _contract(guarantee, assumption=None, desc=None) -> Contract:
    return Contract(agent=AGENT, guarantee=guarantee, assumption=assumption, desc=desc)


# ---------------------------------------------------------------------------
# End-to-end: compatibility, then conflict-freedom
# ---------------------------------------------------------------------------


class TestCheckConflicts:
    def test_conflict_free_library(self):
        report = check_conflicts(
            [
                _contract(G(Not(_called("rm"))), desc="never rm"),
                _contract(F(_called("audit")), desc="audit eventually"),
            ],
            backend="native",
        )
        assert report.ok
        assert report.conflicts == []
        assert report.checked == 2

    def test_conflict_under_assumptions_that_hold_together(self):
        # Both assume a clean environment; their guarantees clash.
        c1 = _contract(
            G(Not(_called("db_write"))),
            assumption=G(Not(_leaks("fetch"))),
            desc="no db writes",
        )
        c2 = _contract(
            F(_called("db_write")),
            assumption=G(Not(_leaks("fetch"))),
            desc="must write db",
        )
        report = check_conflicts([c1, c2], backend="native")
        assert not report.ok
        assert len(report.conflicts) == 1
        assert set(report.conflicts[0].labels) == {"no db writes", "must write db"}

    def test_assumptions_that_clash_are_caught_by_the_same_query(self):
        # One contract requires the environment to always leak, the
        # other requires it never to. No session satisfies both, so the
        # joint query catches it without a separate assumption check.
        c1 = _contract(
            F(_called("y")), assumption=G(_leaks("fetch")), desc="wants a leak"
        )
        c2 = _contract(
            F(_called("y")), assumption=G(Not(_leaks("fetch"))), desc="bans a leak"
        )
        report = check_conflicts([c1, c2], backend="native")
        assert not report.ok
        assert set(report.conflicts[0].labels) == {"wants a leak", "bans a leak"}

    def test_unconditional_contracts_are_checked_together(self):
        c1 = _contract(G(Not(_called("w"))), desc="never w")
        c2 = _contract(F(_called("w")), desc="eventually w")
        report = check_conflicts([c1, c2], backend="native")
        assert len(report.conflicts) == 1

    def test_core_is_minimal_third_contract_excluded(self):
        c1 = _contract(G(Not(_called("w"))), desc="never w")
        c2 = _contract(F(_called("w")), desc="eventually w")
        c3 = _contract(F(_called("log")), desc="unrelated")
        report = check_conflicts([c1, c2, c3], backend="native")
        assert len(report.conflicts) == 1
        assert set(report.conflicts[0].labels) == {"never w", "eventually w"}

    def test_two_disjoint_conflicts(self):
        report = check_conflicts(
            [
                _contract(G(Not(_called("a"))), desc="never a"),
                _contract(F(_called("a")), desc="eventually a"),
                _contract(G(Not(_called("b"))), desc="never b"),
                _contract(F(_called("b")), desc="eventually b"),
            ],
            backend="native",
        )
        assert len(report.conflicts) == 2

    def test_sto_contracts_are_skipped(self):
        sto = Atom("helpful", atom_type="sto", output_type="classify")
        report = check_conflicts(
            [
                _contract(G(sto), desc="sto rule"),
                _contract(F(_called("x")), desc="det rule"),
            ],
            backend="native",
        )
        assert report.skipped == ["sto rule"]
        assert report.checked == 1

    def test_fewer_than_two_contracts_is_trivially_ok(self):
        report = check_conflicts(
            [_contract(And(_called("a"), Not(_called("a"))), desc="broken")],
            backend="native",
        )
        assert report.ok  # nothing to conflict *with*; consistency is separate

    def test_bad_backend_rejected(self):
        with pytest.raises(ValueError):
            check_conflicts([], backend="quantum")


# ---------------------------------------------------------------------------
# MUC extraction
# ---------------------------------------------------------------------------


class TestExtractMuc:
    def test_shrinks_to_the_clashing_pair(self):
        contracts = [
            _contract(F(_called("log")), desc="u0"),
            _contract(G(Not(_called("w"))), desc="u1"),
            _contract(F(_called("audit")), desc="u2"),
            _contract(F(_called("w")), desc="u3"),
        ]
        units, _ = _units_of(contracts)

        from contragent.formulas.sat import is_satisfiable

        def sat_of(subset):
            return is_satisfiable([u.combined for u in subset])

        assert sat_of(units) is False
        core, minimal = extract_muc(units, sat_of)
        assert minimal
        assert sorted(u.label for u in core) == ["u1", "u3"]


# ---------------------------------------------------------------------------
# Domain-constraint derivation
# ---------------------------------------------------------------------------


class TestDomainConstraints:
    def test_called_atoms_are_mutex(self):
        formulas = [G(Not(_called("a"))), F(_called("b"))]
        mutex, impl, axioms = derive_domain_constraints(formulas)
        assert mutex == [["called(a)", "called(b)"]]
        assert impl == []
        assert axioms == []

    def test_tool_scoped_predicates_imply_called(self):
        cw = Atom("called_with", "x", "rm")
        formulas = [F(cw), G(Not(_called("x")))]
        _, impl, _ = derive_domain_constraints(formulas)
        assert ("called_with(x, rm)", "called(x)") in impl

    def test_count_ban_axiom_emitted(self):
        from tests._builders import must_precede, rate_limit

        formulas = [
            rate_limit("check", 0).formula,
            must_precede("check", "pay").formula,
        ]
        _, _, axioms = derive_domain_constraints(formulas)
        # Ban axiom G(called(check) -> !(count <= 0)) plus the monotone
        # persistence axiom G(!(count <= 0) -> G(!(count <= 0))).
        assert len(axioms) == 2

    def test_numeric_bound_conflict_detected(self):
        # rate_limit(x, 3) vs "x must reach 10 calls": the pointwise
        # theory checker relates count<=3 and count>=10 over one
        # register — the pure boolean abstraction cannot see this.
        from contragent.formulas.formula import Const, Ge, Var
        from tests._builders import rate_limit

        contracts = [
            _contract(rate_limit("x", 3), desc="at most 3"),
            _contract(F(Ge(Var("count", "x"), Const(10))), desc="needs 10"),
        ]
        report = check_conflicts(contracts, backend="native")
        assert len(report.conflicts) == 1
        assert set(report.conflicts[0].labels) == {"at most 3", "needs 10"}
        # Escape hatch: with theory reasoning and exact counters both
        # off, the pure boolean abstraction cannot see the clash.
        report_bare = check_conflicts(
            contracts, backend="native", theory_checker=None, exact_counters=False
        )
        assert report_bare.conflicts == []

    def test_monotone_persistence_axiom(self):
        # "count reaches 2, then later drops to <= 1" is impossible for
        # a monotone register. Pointwise theory alone cannot see it (the
        # two comparisons hold at *different* events); the persistence
        # axiom closes the gap for gadget-less queries, and the native
        # gadget catches it outright.
        from contragent.formulas.formula import And as FAnd
        from contragent.formulas.formula import Const, Ge, Le, Var
        from contragent.formulas.sat import is_satisfiable

        x = Var("count", "x")
        formula = F(FAnd(Ge(x, Const(2)), F(Le(x, Const(1)))))
        # Pointwise alone (no gadget): SAT — the gap this axiom closes.
        assert is_satisfiable(formula, exact_counters=False) is True
        _, _, axioms = derive_domain_constraints([formula])
        assert any("count" in repr(a) for a in axioms)
        assert is_satisfiable([formula, *axioms], exact_counters=False) is False
        # The saturating-counter gadget needs no axiom at all.
        assert is_satisfiable(formula) is False

    def test_cross_vocabulary_conflict_end_to_end(self):
        # The former blind spot: the bound speaks `count`, the forcing
        # speaks `called`. The saturating-counter gadget links them.
        from contragent.formulas.formula import X
        from tests._builders import rate_limit

        cx = _called("x")
        four_calls = F(And(cx, X(F(And(cx, X(F(And(cx, X(F(cx))))))))))
        contracts = [
            _contract(rate_limit("x", 3), desc="at most 3"),
            _contract(four_calls, desc="four call events"),
        ]
        report = check_conflicts(contracts, backend="native")
        assert len(report.conflicts) == 1
        assert set(report.conflicts[0].labels) == {"at most 3", "four call events"}

    def test_counting_axioms_match_gadget(self):
        # The LTLf counting axioms (mus2muc's propositional stand-in for
        # the native gadget) must agree with the gadget's verdicts.
        from contragent.analysis.conflicts import counting_axioms
        from contragent.formulas.formula import Const, Ge, Le, Var, X
        from contragent.formulas.sat import is_satisfiable

        x = Var("count", "x")
        cx = _called("x")
        four_calls = F(And(cx, X(F(And(cx, X(F(And(cx, X(F(cx))))))))))
        for formulas in (
            [G(Le(x, Const(3))), four_calls],
            [F(Ge(x, Const(4))), G(Not(cx))],
            [F(Ge(x, Const(4)))],
            [G(Le(x, Const(0))), F(cx)],
        ):
            gadget = is_satisfiable(formulas)
            axioms = counting_axioms(formulas)
            assert (
                is_satisfiable(list(formulas) + axioms, exact_counters=False) is gadget
            )

    def test_virtual_driver_joins_mutex(self):
        # count(x) comparisons pull a virtual called(x) atom into the
        # mutex group even though no formula mentions it.
        from contragent.formulas.formula import Const, Le, Var

        formulas = [G(Le(Var("count", "x"), Const(3))), F(_called("y"))]
        mutex, _, _ = derive_domain_constraints(formulas)
        assert mutex == [["called(x)", "called(y)"]]

    def test_rate_limit_ban_conflict_detected(self):
        # The README-style clash: a refund requires a prior policy
        # check, policy checks are banned via rate_limit(0), and a
        # refund must eventually happen. Only the count-ban axiom
        # (called(t) => count(t) >= 1) links the pieces.
        from tests._builders import must_precede, rate_limit

        report = check_conflicts(
            [
                _contract(must_precede("check_policy", "issue_refund")),
                _contract(rate_limit("check_policy", 0), desc="checks banned"),
                _contract(F(_called("issue_refund")), desc="must refund"),
            ],
            backend="native",
        )
        assert len(report.conflicts) == 1
        assert len(report.conflicts[0].units) == 3

    def test_constraints_expose_conflict(self):
        # One tool call per event: "always calls a" + "eventually calls b"
        # is only detectable as a conflict with the mutex constraint.
        report = check_conflicts(
            [
                _contract(G(_called("a")), desc="always a"),
                _contract(F(_called("b")), desc="eventually b"),
            ],
            backend="native",
        )
        assert len(report.conflicts) == 1


# ---------------------------------------------------------------------------
# mus2muc backend: serializer + graceful unavailability
# ---------------------------------------------------------------------------


class TestMus2mucBackend:
    def test_serializer_output_shape(self):
        contracts = [
            _contract(G(Not(_called("a"))), desc="never a"),
            _contract(
                And(F(_called("a")), F(_called("b"))),
                assumption=G(Not(_leaks("fetch"))),
                desc="wants a",
            ),
        ]
        units, _ = _units_of(contracts)
        text, mapping = units_to_ltlfconj(
            units, mutex_groups=[["called(a)", "called(b)"]]
        )
        assert set(mapping) == {"c0", "c1"}
        lines = [ln for ln in text.splitlines() if ln]
        assert all(ln.endswith(";") for ln in lines)
        assert any(ln.startswith("c0 := G(") for ln in lines)
        # Domain mutex constraint emitted as its own labeled conjunct.
        assert any(ln.startswith("dom0 := G(!(") for ln in lines)
        # Only generated lowercase symbols — raw pred keys never leak.
        assert "called(" not in text

    def test_serializer_propositionalizes_theory(self):
        # mus2muc is purely propositional, so pointwise arithmetic facts
        # must be emitted as dom conjuncts. For canonicalized total
        # comparisons "x<=3 excludes x>=10" surfaces as the implication
        # G(p_le3 -> p_lt10) (x>=10 is the negated lt-proposition);
        # partial-term orientations get an explicit G-mutex.
        from contragent.formulas.formula import ArgValue, Const, Ge, Gt, Le, Var
        from contragent.formulas.theory import default_theory_checker

        x = Var("count", "x")
        amt = ArgValue("transfer", "amount")
        contracts = [
            _contract(G(Le(x, Const(3))), desc="cap 3"),
            _contract(F(Ge(x, Const(10))), desc="reach 10"),
            _contract(F(Le(amt, Const(5))), desc="amt small"),
            _contract(F(Gt(amt, Const(5))), desc="amt big"),
        ]
        units, _ = _units_of(contracts)
        text, _ = units_to_ltlfconj(units, theory_checker=default_theory_checker())
        dom_lines = [ln for ln in text.splitlines() if ln.startswith("dom")]
        assert any("->" in ln for ln in dom_lines)  # total: implication
        assert any("!(" in ln and "&" in ln for ln in dom_lines)  # partial: mutex

    def test_unavailable_reason_mentions_install(self, tmp_path):
        reason = unavailable_reason(bin_folder=tmp_path)
        assert "wasp" in reason
        assert "github.com/ainnoot/mus2muc" in reason

    def test_explicit_mus2muc_backend_raises_when_missing(self, tmp_path):
        if is_available(bin_folder=tmp_path):  # pragma: no cover
            pytest.skip("mus2muc toolchain unexpectedly present")
        with pytest.raises(Mus2mucUnavailable):
            check_conflicts(
                [
                    _contract(G(Not(_called("w"))), desc="never w"),
                    _contract(F(_called("w")), desc="eventually w"),
                ],
                backend="mus2muc",
                mus2muc_bin_folder=tmp_path,
            )

    def test_auto_falls_back_to_native(self, tmp_path):
        report = check_conflicts(
            [
                _contract(G(Not(_called("w"))), desc="never w"),
                _contract(F(_called("w")), desc="eventually w"),
            ],
            backend="auto",
            mus2muc_bin_folder=tmp_path,
        )
        assert report.backend == "native"
        assert len(report.conflicts) == 1


# ---------------------------------------------------------------------------
# Witness-trace fast path (searchless SAT certificate)
# ---------------------------------------------------------------------------


class TestWitnessCertificate:
    def _all_safety_library(self, n=30):
        # n safety contracts, each with its own called atom AND its own
        # comparison proposition — the alphabet estimate (mutex group of
        # n call atoms x 2^n free comparison props) blows the default
        # max_alphabet instantly, so the automata search alone returns
        # unknown on the full conjunction.
        from contragent.formulas.formula import ArgValue, Const, Implies, Le

        return [
            _contract(
                G(
                    Implies(
                        _called(f"tool_{i}"),
                        Le(ArgValue(f"tool_{i}", "amount"), Const(100 + i)),
                    )
                ),
                desc=f"cap amount on tool_{i}",
            )
            for i in range(n)
        ]

    def test_large_all_safety_library_certified_searchlessly(self):
        import time

        from contragent.formulas.sat import is_satisfiable

        contracts = self._all_safety_library()
        # Premise: the search alone cannot decide this library — the
        # alphabet budget is exceeded before any state is explored.
        units, _ = _units_of(contracts)
        assert is_satisfiable([u.combined for u in units]) is None
        # The all-false witness satisfies every safety contract, so the
        # check is conflict-free without touching the search budgets.
        start = time.monotonic()
        report = check_conflicts(contracts, backend="native")
        elapsed = time.monotonic() - start
        assert report.ok
        assert not report.unknown
        assert report.certificate == "witness-trace"
        assert elapsed < 1.0

    def test_conflict_still_reported_with_search_certificate(self):
        report = check_conflicts(
            [
                _contract(G(Not(_called("w"))), desc="never w"),
                _contract(F(_called("w")), desc="eventually w"),
            ],
            backend="native",
        )
        assert not report.ok
        assert len(report.conflicts) == 1
        assert report.certificate == "search"

    def test_f_obligation_satisfied_by_single_call_witness(self):
        # The all-false witness fails F(called(x)) — weak finite-trace
        # semantics, matching the sat engine — but the single
        # called(x) witness satisfies both contracts, so the verdict
        # is still searchless.
        report = check_conflicts(
            [
                _contract(F(_called("x")), desc="eventually x"),
                _contract(G(Not(_called("rm"))), desc="never rm"),
            ],
            backend="native",
        )
        assert report.ok
        assert report.certificate == "witness-trace"

    def test_no_short_witness_falls_through_to_search(self):
        # Two eventualities over distinct tools need a length-2 trace
        # (one call per event); every length-1 witness fails one of
        # them, so the verdict comes from the search — and is still
        # conflict-free.
        report = check_conflicts(
            [
                _contract(F(_called("a")), desc="eventually a"),
                _contract(F(_called("b")), desc="eventually b"),
            ],
            backend="native",
        )
        assert report.ok
        assert report.conflicts == []
        assert report.certificate == "search"

    def test_witness_counter_semantics_match_grounding(self):
        # A single-called(x) witness carries count(x) = 1 (grounding
        # increments before emitting), so it must NOT certify a library
        # where x is both required and rate-limited to zero — the
        # conflict has to survive the fast path.
        from tests._builders import rate_limit

        report = check_conflicts(
            [
                _contract(rate_limit("x", 0), desc="x banned"),
                _contract(F(_called("x")), desc="eventually x"),
            ],
            backend="native",
        )
        assert not report.ok
        assert len(report.conflicts) == 1
        assert report.certificate == "search"

    def test_certificate_defaults_none(self):
        from contragent.analysis import ConflictReport

        report = ConflictReport()
        assert report.certificate is None
        # Trivial libraries (fewer than two det units) carry no
        # certificate either — nothing was proved.
        trivial = check_conflicts(
            [_contract(G(Not(_called("a"))), desc="solo")], backend="native"
        )
        assert trivial.ok
        assert trivial.certificate is None

    def test_render_mentions_witness_certificate(self):
        report = check_conflicts(self._all_safety_library(n=3), backend="native")
        assert report.certificate == "witness-trace"
        text = report.render()
        assert "conflict-free" in text
        assert "witness-trace" in text


# ---------------------------------------------------------------------------
# ContrAgent integration
# ---------------------------------------------------------------------------


class TestGuardIntegration:
    def _guard(self, **kwargs):
        from contragent.core import ContrAgent
        from tests._builders import must_precede

        return ContrAgent(
            agent_id="bot",
            contracts=[
                {"guarantee": must_precede("check_policy", "issue_refund")},
                {"guarantee": G(Not(_called("check_policy"))), "desc": "no checks"},
                {"guarantee": F(_called("issue_refund")), "desc": "must refund"},
            ],
            mode="observe",
            **kwargs,
        )

    def test_check_conflicts_method(self):
        report = self._guard().check_conflicts(backend="native")
        assert len(report.conflicts) == 1
        # All three contracts participate: refund required, checks
        # banned, but a check must precede any refund.
        assert len(report.conflicts[0].units) == 3

    def test_init_kwarg_warns_but_loads(self, capsys):
        guard = self._guard(conflict_check=True)
        err = capsys.readouterr().err
        assert "conflict" in err
        # The guard must still construct and carry its contracts — a
        # conflicted library warns, it does not fail to load.
        assert len(guard._system._contracts) == 3

    def test_report_render_smoke(self):
        report = self._guard().check_conflicts(backend="native")
        text = report.render()
        assert "conflict" in text
