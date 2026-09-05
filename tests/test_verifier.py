"""Standalone tests for the TraceVerifier — pure formal evaluation without monitor.

Exercises:
  - Verdict / ContractVerdict shapes
  - check() on single formulas
  - check_contract() assumption short-circuit + enforcement list
  - Incremental grounding correctness across multiple sync() calls
  - Incremental eval correctness (G-cache hit/miss/rollback)
  - Trace shrinkage auto-reset
  - Standalone use (no monitor) with Contract objects
"""

from __future__ import annotations

from contragent.models.agent import Agent
from contragent.models.contract import Contract
from contragent.models.trace import Event, Trace
from tests._builders import (
    arg_blacklist,
    must_precede,
    no_reversal,
    rate_limit,
)
from contragent.runtime.verifier import ContractVerdict, TraceVerifier, Verdict


def _trace(*tool_calls: str) -> Trace:
    """Build a trace of tool-only events for the given tool name sequence."""
    return Trace(
        events=[
            Event(ts=i, agent="bot", event_type="tool_call", tool=t)
            for i, t in enumerate(tool_calls)
        ]
    )


class TestVerdict:
    def test_verdict_is_truthy(self):
        v = Verdict(holds=True, desc="ok")
        assert bool(v) is True

    def test_verdict_is_falsy(self):
        v = Verdict(holds=False, desc="nope")
        assert bool(v) is False

    def test_contract_verdict_bool_reflects_holds(self):
        cv = ContractVerdict(
            assumptions=[Verdict(holds=True, desc="a")],
            guarantees=[Verdict(holds=True, desc="e")],
        )
        assert bool(cv) is True

    def test_contract_verdict_false_on_enforcement_violation(self):
        cv = ContractVerdict(
            assumptions=[Verdict(holds=True, desc="a")],
            guarantees=[Verdict(holds=False, desc="e")],
        )
        assert bool(cv) is False
        assert len(cv.guarantee_violations) == 1

    def test_contract_verdict_false_on_assumption_failure(self):
        cv = ContractVerdict(
            assumptions=[Verdict(holds=False, desc="a")],
            guarantees=[],
        )
        assert bool(cv) is False
        assert cv.first_assumption_failure is not None
        assert cv.first_assumption_failure.desc == "a"


class TestCheck:
    def test_must_precede_satisfied(self):
        v = TraceVerifier()
        v.sync(_trace("verify", "transfer"))
        result = v.check(must_precede("verify", "transfer"))
        assert result.holds is True

    def test_must_precede_violated(self):
        v = TraceVerifier()
        v.sync(_trace("transfer"))
        result = v.check(must_precede("verify", "transfer"))
        assert result.holds is False

    def test_rate_limit_satisfied(self):
        v = TraceVerifier()
        v.sync(_trace("X", "X"))
        assert v.check(rate_limit("X", 3)).holds is True

    def test_rate_limit_violated(self):
        v = TraceVerifier()
        v.sync(_trace("X", "X", "X", "X"))
        assert v.check(rate_limit("X", 3)).holds is False

    def test_verdict_carries_formula_and_desc(self):
        v = TraceVerifier()
        v.sync(_trace("A", "B"))
        f = must_precede("A", "B")
        result = v.check(f)
        assert result.desc == "A must precede B"
        assert result.formula is f


class TestCheckContract:
    def test_unconditional_contract_passes(self):
        contract = Contract(
            agent=Agent(id="bot"),
            guarantee=rate_limit("X", 3),
        )
        v = TraceVerifier()
        v.sync(_trace("X", "X"))
        cv = v.check_contract(contract)
        assert cv.holds is True
        assert len(cv.assumptions) == 0
        assert len(cv.guarantees) == 1
        assert cv.guarantees[0].holds is True

    def test_assumption_failure_skips_enforcements(self):
        contract = Contract(
            agent=Agent(id="bot"),
            assumption=must_precede("auth", "act"),
            guarantee=rate_limit("act", 3),
        )
        v = TraceVerifier()
        v.sync(_trace("act"))  # auth never called → assumption fails
        cv = v.check_contract(contract)
        assert cv.assumption_holds is False
        assert cv.first_assumption_failure is not None
        assert len(cv.guarantees) == 0  # skipped

    def test_assumption_and_enforcement_both_evaluated_when_a_holds(self):
        contract = Contract(
            agent=Agent(id="bot"),
            assumption=must_precede("auth", "act"),
            guarantee=rate_limit("act", 1),  # will be violated
        )
        v = TraceVerifier()
        v.sync(_trace("auth", "act", "act"))
        cv = v.check_contract(contract)
        assert cv.assumption_holds is True
        assert cv.holds is False
        assert cv.guarantees[0].holds is False

    def test_list_valued_enforcement_all_checked(self):
        contract = Contract(
            agent=Agent(id="bot"),
            guarantee=[rate_limit("X", 3), rate_limit("Y", 3)],
        )
        v = TraceVerifier()
        v.sync(_trace("X", "X", "Y", "Y"))
        cv = v.check_contract(contract)
        assert len(cv.guarantees) == 2
        assert cv.holds is True


class TestIncrementalGrounding:
    def test_sync_appends_only_new_events(self):
        v = TraceVerifier()
        v.sync(_trace("a", "b"))
        assert len(v.valuations) == 2
        assert v._grounded_upto == 2

        # Extend the trace; sync should only ground the new event.
        v.sync(_trace("a", "b", "c"))
        assert len(v.valuations) == 3
        assert v._grounded_upto == 3

    def test_sync_auto_resets_on_trace_shrink(self):
        v = TraceVerifier()
        v.sync(_trace("a", "b", "c"))
        assert len(v.valuations) == 3

        # Shrink to 1 event — should trigger a full reset + re-ground.
        v.sync(_trace("d"))
        assert len(v.valuations) == 1
        assert v.valuations[0].get("called(d)") is True
        # The stale "called(a/b/c)" entries should NOT leak.
        for key in v.valuations[0]:
            assert "called(a)" not in key
            assert "called(b)" not in key
            assert "called(c)" not in key

    def test_sync_auto_resets_when_content_atoms_change(self):
        v = TraceVerifier()
        v.sync(_trace("X"), content_atoms=None)
        first_valuations = list(v.valuations)

        # Different content_atoms → full re-ground.
        v.sync(_trace("X"), content_atoms={"arg_has": {("X", "pattern")}})
        assert len(v.valuations) == 1
        assert v.valuations is not first_valuations

    def test_incremental_count_matches_batch(self):
        """Incremental syncs must produce the same count values as a single batch sync."""
        full_trace = _trace("X", "X", "X", "Y", "X")

        # Batch: sync once at the end
        v_batch = TraceVerifier()
        v_batch.sync(full_trace)

        # Incremental: sync after each append
        v_inc = TraceVerifier()
        for n in range(1, len(full_trace.events) + 1):
            partial = Trace(events=full_trace.events[:n])
            v_inc.sync(partial)

        assert len(v_batch.valuations) == len(v_inc.valuations)
        for a, b in zip(v_batch.valuations, v_inc.valuations):
            assert a == b


class TestIncrementalEval:
    def test_g_cache_hits_on_stable_true(self):
        """Re-evaluating the same G-rooted formula should use the cache."""
        v = TraceVerifier()
        f = rate_limit("X", 1_000_000)
        v.sync(_trace("X"))
        assert v.check(f).holds is True

        v.sync(_trace("X", "X"))
        # Cache was {(scanned=1, True)}; we should only scan position 1.
        assert v.check(f).holds is True
        # Sanity: scanned_upto is at the latest length
        raw = f.formula
        assert v._g_cache[raw] == (2, True)

    def test_g_cache_transitions_to_false_and_sticks(self):
        v = TraceVerifier()
        f = rate_limit("X", 2)
        v.sync(_trace("X"))
        assert v.check(f).holds is True

        v.sync(_trace("X", "X"))
        assert v.check(f).holds is True

        v.sync(_trace("X", "X", "X"))
        assert v.check(f).holds is False

        # Subsequent calls should still report False from the cache.
        assert v.check(f).holds is False

    def test_nested_temporal_falls_through_to_full_eval(self):
        """no_reversal contains nested G and must not be cached prematurely."""
        v = TraceVerifier()
        f = no_reversal("approve", "deny")

        v.sync(_trace("approve"))
        assert v.check(f).holds is True

        v.sync(_trace("approve", "deny"))
        # This is the bug we fixed: caching the outer G at len=1 as
        # True and then only checking position 1 would miss that the
        # inner G's scope extended.
        assert v.check(f).holds is False

    def test_reset_clears_g_cache(self):
        v = TraceVerifier()
        f = rate_limit("X", 1)
        v.sync(_trace("X", "X"))
        assert v.check(f).holds is False  # violated

        v.reset()
        v.sync(_trace("X"))
        assert v.check(f).holds is True  # clean slate


class TestCheckAssumption:
    def test_check_assumption_unconditional(self):
        contract = Contract(
            agent=Agent(id="bot"),
            guarantee=rate_limit("X", 3),
        )
        v = TraceVerifier()
        v.sync(_trace("X"))
        result = v.check_assumption(contract)
        assert result.holds is True
        assert result.desc == "true"

    def test_check_assumption_fails_short_circuit(self):
        contract = Contract(
            agent=Agent(id="bot"),
            assumption=[
                must_precede("a", "b"),
                must_precede("c", "d"),  # should not reach here
            ],
            guarantee=rate_limit("X", 3),
        )
        v = TraceVerifier()
        v.sync(_trace("b"))  # first assumption fails
        result = v.check_assumption(contract)
        assert result.holds is False
        assert "a must precede b" in result.desc.lower()


class TestAgainstMonitor:
    """Integration check: TraceVerifier used standalone on the same trace
    as the monitor should return the same verdict."""

    def test_verifier_matches_monitor_on_rate_limit(self):
        import contragent

        guard = contragent.ContrAgent(
            agent_id="bot",
            contracts=["G((Var('count', 'X') <= 2))"],
        )
        guard.guard_before("X")
        guard.guard_before("X")  # at limit
        # Using verifier directly
        v = guard.supervisor.verifier
        result = v.check(rate_limit("X", 2))
        assert result.holds is True

        # One more — monitor would block; but verifier sees the trace
        # state and still reports "at limit". The blocked event is
        # popped by guard, so verifier should still see the old state.
        blocked = guard.guard_before("X")
        assert blocked.blocked is True
        # TraceVerifier re-evaluated correctly after rollback reset.
        assert v.check(rate_limit("X", 2)).holds is True

    def test_verifier_accessible_from_monitor_property(self):
        import contragent

        guard = contragent.ContrAgent(
            agent_id="bot",
            contracts=["G((Var('count', 'X') <= 2))"],
        )
        assert isinstance(guard.supervisor.verifier, TraceVerifier)


class TestBackCompatAlias:
    def test_verifier_alias_still_works(self):
        """The original ``Verifier`` name is kept as a deprecated alias."""
        from contragent.runtime.verifier import TraceVerifier, Verifier

        assert Verifier is TraceVerifier


class TestArgBlacklistCacheable:
    """arg_blacklist = G(Not(arg_field_has(...))) — flat G, should cache."""

    def test_arg_blacklist_benefits_from_cache(self):
        v = TraceVerifier()
        f = arg_blacklist("bash", "cmd", ["rm -rf"])

        trace = Trace(
            events=[
                Event(
                    ts=0,
                    agent="bot",
                    event_type="tool_call",
                    tool="bash",
                    args={"cmd": "ls /tmp"},
                ),
                Event(
                    ts=1,
                    agent="bot",
                    event_type="tool_call",
                    tool="bash",
                    args={"cmd": "cat foo.txt"},
                ),
            ]
        )
        v.sync(trace, content_atoms={"arg_field_has": {("bash", "cmd", "rm -rf")}})
        assert v.check(f).holds is True
        # G-cache should have registered this formula
        raw = f.formula
        assert raw in v._g_cache
