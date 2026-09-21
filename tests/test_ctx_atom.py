"""Tests for the ``ctx(k, v)`` / ``ctx_matches(k, pattern)`` atom family
and the ``observe_context`` hook that populates them.

These atoms are the runtime bridge between ContrAgent and the host stack's
identity / provenance / trust systems — they upgrade ASI-03 / ASI-06 /
ASI-07 coverage from Partial to a fully-enforceable contract layer,
provided the integration feeds the relevant facts in.
"""

from __future__ import annotations

from contragent.formulas._pred_key import pred_key
from contragent.formulas.evaluator import evaluate
from contragent.formulas.formula import Atom
from contragent.models.trace import Event, Trace
from contragent.tracer.grounding import GroundingState, collect_content_atoms, ground
from tests._helpers import ltl


def _trace(*events: Event) -> Trace:
    return Trace(events=list(events))


def _ctx_update(ts: int, agent: str, facts: dict[str, str]) -> Event:
    return Event(
        ts=ts,
        agent=agent,
        event_type="context_update",
        args=facts,
    )


def _tool(ts: int, agent: str, tool: str, args: dict | None = None) -> Event:
    return Event(ts=ts, agent=agent, event_type="tool_call", tool=tool, args=args)


# ---------------------------------------------------------------------------
# ctx(k, v) — basic emission
# ---------------------------------------------------------------------------


def test_context_update_emits_ctx_atoms_at_same_timestep():
    """``ctx(k, v)`` is visible at the same event that set it — matches
    the intuition that after ``observe_context({...})`` the facts are
    immediately active for any contract evaluated at that step."""
    trace = _trace(
        _ctx_update(0, "bot", {"caller_id": "alice", "source": "canonical:/v3"}),
    )
    valuations = ground(trace)
    v0 = valuations[0]

    assert v0.get(pred_key("ctx", "caller_id", "alice")) is True
    assert v0.get(pred_key("ctx", "source", "canonical:/v3")) is True


def test_ctx_persists_across_events_until_overridden():
    """Context is sticky. Set once, every subsequent event carries it
    until the same key is overwritten by a later update."""
    trace = _trace(
        _ctx_update(0, "bot", {"caller_id": "alice"}),
        _tool(1, "bot", "read_file"),
        _tool(2, "bot", "write_file"),
        _ctx_update(3, "bot", {"caller_id": "bob"}),
        _tool(4, "bot", "write_file"),
    )
    valuations = ground(trace)

    # alice is the caller for ts 0–2
    for ts in (0, 1, 2):
        assert valuations[ts].get(pred_key("ctx", "caller_id", "alice")) is True
        assert pred_key("ctx", "caller_id", "bob") not in valuations[ts]

    # bob takes over at ts 3
    assert valuations[3].get(pred_key("ctx", "caller_id", "bob")) is True
    assert pred_key("ctx", "caller_id", "alice") not in valuations[3]
    assert valuations[4].get(pred_key("ctx", "caller_id", "bob")) is True


def test_ctx_merge_does_not_clear_unrelated_keys():
    """Later updates merge — they should not wipe keys they don't
    mention. This is critical for integrations that push caller_id on
    every request but content_source only on retrieval events."""
    trace = _trace(
        _ctx_update(0, "bot", {"caller_id": "alice", "tenant": "acme"}),
        _ctx_update(1, "bot", {"source": "canonical:/v3"}),
        _tool(2, "bot", "answer"),
    )
    valuations = ground(trace)
    v2 = valuations[2]

    assert v2.get(pred_key("ctx", "caller_id", "alice")) is True
    assert v2.get(pred_key("ctx", "tenant", "acme")) is True
    assert v2.get(pred_key("ctx", "source", "canonical:/v3")) is True


def test_ctx_empty_update_is_noop():
    """Edge case — ``observe_context({})`` shouldn't break anything or
    clobber existing context. In practice the hook filters None values
    before emitting, so this is defense-in-depth at the grounding layer."""
    trace = _trace(
        _ctx_update(0, "bot", {"caller_id": "alice"}),
        Event(ts=1, agent="bot", event_type="context_update", args={}),
        _tool(2, "bot", "read"),
    )
    valuations = ground(trace)
    assert valuations[2].get(pred_key("ctx", "caller_id", "alice")) is True


# ---------------------------------------------------------------------------
# ctx_matches(k, pattern) — regex
# ---------------------------------------------------------------------------


def test_ctx_matches_emitted_only_when_in_content_atoms():
    """``ctx_matches`` is a content atom — the grounding layer only
    evaluates (key, pattern) tuples that appear in some formula.
    Without a formula using it, the atom is absent."""
    trace = _trace(
        _ctx_update(0, "bot", {"caller_id": "spiffe://prod/ap-agent"}),
        _tool(1, "bot", "wire_transfer"),
    )
    # No content_atoms → ctx_matches not evaluated
    valuations = ground(trace)
    for v in valuations:
        for key in v:
            assert not key.startswith("ctx_matches(")


def test_ctx_matches_regex_against_current_value():
    formula = Atom("ctx_matches", "caller_id", r"^spiffe://prod/.*")
    content_atoms = collect_content_atoms([formula])

    trace = _trace(
        _ctx_update(0, "bot", {"caller_id": "spiffe://prod/ap-agent"}),
        _tool(1, "bot", "wire_transfer"),
    )
    state = GroundingState()
    v0 = None
    v1 = None
    for idx, event in enumerate(trace.events):
        from contragent.tracer.grounding import ground_event

        val = ground_event(event, idx, state, content_atoms=content_atoms)
        if idx == 0:
            v0 = val
        else:
            v1 = val

    # Both timesteps see the matching caller_id — ctx_matches fires at both
    assert v0[pred_key("ctx_matches", "caller_id", r"^spiffe://prod/.*")] is True
    assert v1[pred_key("ctx_matches", "caller_id", r"^spiffe://prod/.*")] is True


def test_ctx_matches_false_when_value_does_not_match():
    formula = Atom("ctx_matches", "caller_id", r"^spiffe://prod/.*")
    content_atoms = collect_content_atoms([formula])

    trace = _trace(
        _ctx_update(0, "bot", {"caller_id": "spiffe://dev/test-agent"}),
        _tool(1, "bot", "wire_transfer"),
    )
    from contragent.tracer.grounding import ground_event

    state = GroundingState()
    valuations = [
        ground_event(e, i, state, content_atoms=content_atoms)
        for i, e in enumerate(trace.events)
    ]
    assert (
        valuations[1][pred_key("ctx_matches", "caller_id", r"^spiffe://prod/.*")]
        is False
    )


def test_ctx_matches_false_when_key_absent():
    formula = Atom("ctx_matches", "caller_id", r"^spiffe://prod/.*")
    content_atoms = collect_content_atoms([formula])

    trace = _trace(_tool(0, "bot", "wire_transfer"))
    from contragent.tracer.grounding import ground_event

    state = GroundingState()
    val = ground_event(trace.events[0], 0, state, content_atoms=content_atoms)
    assert val[pred_key("ctx_matches", "caller_id", r"^spiffe://prod/.*")] is False


# ---------------------------------------------------------------------------
# ctx(k, v) gate on a tool call — end-to-end contract check
# ---------------------------------------------------------------------------


def test_ctx_gate_allows_when_ctx_holds_an_allowed_value():
    # Shipped string: contragent/contracts/benchmark/tau2_bench.yaml (airline)
    det = ltl("G(called(send_certificate) -> ((ctx(user_membership, silver) | ctx(user_membership, gold)) | ctx(user_membership, premium)))")
    trace = _trace(
        _ctx_update(0, "bot", {"user_membership": "silver"}),
        _tool(1, "bot", "send_certificate"),
    )
    valuations = ground(trace)
    assert evaluate(det.formula, valuations) is True


def test_ctx_gate_blocks_when_ctx_is_missing():
    """Fail-closed: if the integration forgot to push the fact at all,
    the contract violates. Loud failure beats silent bypass."""
    # Shipped string: contragent/contracts/benchmark/tau2_bench.yaml (retail)
    det = ltl("G(called(cancel_pending_order) -> ctx(user_authenticated, yes))")
    trace = _trace(_tool(0, "bot", "cancel_pending_order"))
    valuations = ground(trace)
    assert evaluate(det.formula, valuations) is False


def test_ctx_gate_blocks_when_ctx_holds_another_value():
    # Shipped string: contragent/contracts/benchmark/tau2_bench.yaml (retail)
    det = ltl("G(called(cancel_pending_order) -> ctx(user_authenticated, yes))")
    trace = _trace(
        _ctx_update(0, "bot", {"user_authenticated": "no"}),
        _tool(1, "bot", "cancel_pending_order"),
    )
    valuations = ground(trace)
    assert evaluate(det.formula, valuations) is False


# ---------------------------------------------------------------------------
# ctx_matches(k, pattern) gate on a tool call — end-to-end contract check
# ---------------------------------------------------------------------------


def test_ctx_regex_gate_allows_pattern_match():
    det = ltl("G((called('wire_transfer') -> ctx_matches('caller_id', '^spiffe://prod/.*')))")
    # ``ctx_matches`` is a content atom — batch ``ground()`` needs the
    # pattern set via ``collect_content_atoms`` to know what to evaluate.
    # This mirrors how ``arg_has`` / ``llm_said`` / ``count_with`` are used.
    content_atoms = collect_content_atoms([det.formula])
    trace = _trace(
        _ctx_update(0, "bot", {"caller_id": "spiffe://prod/ap-agent"}),
        _tool(1, "bot", "wire_transfer"),
    )
    valuations = ground(trace, content_atoms=content_atoms)
    assert evaluate(det.formula, valuations) is True


def test_ctx_regex_gate_blocks_pattern_mismatch():
    det = ltl("G((called('wire_transfer') -> ctx_matches('caller_id', '^spiffe://prod/.*')))")
    content_atoms = collect_content_atoms([det.formula])
    trace = _trace(
        _ctx_update(0, "bot", {"caller_id": "spiffe://dev/test"}),
        _tool(1, "bot", "wire_transfer"),
    )
    valuations = ground(trace, content_atoms=content_atoms)
    assert evaluate(det.formula, valuations) is False


# ---------------------------------------------------------------------------
# Combination with other formulas — ctx composes cleanly with an
# ordering rule (weak until) / argument bounds / etc. Proves ctx fits
# into the contract language without special-casing.
# ---------------------------------------------------------------------------


def test_ctx_composes_with_ordering_contract():
    """Combine the shipped ordering rule (get_order_details before
    cancel_pending_order) with the shipped identity gate
    (cancel_pending_order only when user_authenticated): cancelling is
    allowed only AFTER the lookup AND when the user is authenticated."""
    # Both strings: contragent/contracts/benchmark/tau2_bench.yaml (retail)
    ordering = ltl("((!(called('cancel_pending_order')) U called('get_order_details')) | G(!(called('cancel_pending_order'))))")
    identity = ltl("G(called(cancel_pending_order) -> ctx(user_authenticated, yes))")

    # Happy path: both satisfied
    trace_ok = _trace(
        _ctx_update(0, "bot", {"user_authenticated": "yes"}),
        _tool(1, "bot", "get_order_details"),
        _tool(2, "bot", "cancel_pending_order"),
    )
    valuations_ok = ground(trace_ok)
    assert evaluate(ordering.formula, valuations_ok) is True
    assert evaluate(identity.formula, valuations_ok) is True

    # Fails ordering (cancel before lookup), identity still holds
    trace_bad_order = _trace(
        _ctx_update(0, "bot", {"user_authenticated": "yes"}),
        _tool(1, "bot", "cancel_pending_order"),
    )
    valuations_bo = ground(trace_bad_order)
    assert evaluate(ordering.formula, valuations_bo) is False
    assert evaluate(identity.formula, valuations_bo) is True

    # Fails identity (user_authenticated=no), ordering still holds
    trace_bad_ctx = _trace(
        _ctx_update(0, "bot", {"user_authenticated": "no"}),
        _tool(1, "bot", "get_order_details"),
        _tool(2, "bot", "cancel_pending_order"),
    )
    valuations_bc = ground(trace_bad_ctx)
    assert evaluate(ordering.formula, valuations_bc) is True
    assert evaluate(identity.formula, valuations_bc) is False


# ---------------------------------------------------------------------------
# GroundingState.reset() — regression guard: ctx must not leak
# across session boundaries in the verifier.
# ---------------------------------------------------------------------------


def test_reset_clears_current_ctx():
    state = GroundingState()
    state.current_ctx["caller_id"] = "alice"
    state.current_ctx["source"] = "canonical:/v3"
    state.reset()
    assert state.current_ctx == {}


# ---------------------------------------------------------------------------
# Non-string values in context facts
# ---------------------------------------------------------------------------


def test_ctx_stringifies_non_string_values():
    """The hook itself allows any value type; grounding stringifies so
    atom keys stay hashable. Bools, ints, floats all become their
    str() form — users should think of ctx keys as strings for
    contract-matching purposes."""
    trace = _trace(
        _ctx_update(0, "bot", {"retry_count": 3, "verified": True, "confidence": 0.95}),
    )
    valuations = ground(trace)
    v0 = valuations[0]
    assert v0.get(pred_key("ctx", "retry_count", "3")) is True
    assert v0.get(pred_key("ctx", "verified", "True")) is True
    assert v0.get(pred_key("ctx", "confidence", "0.95")) is True
