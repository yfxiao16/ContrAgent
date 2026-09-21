"""Comprehensive coverage — grounding-layer atoms (Python).

Grounds events directly via ``ground_event`` so each atom's emission
contract can be asserted in isolation. Mirrors
``ts/packages/sdk/src/__tests__/comprehensive_atoms.test.ts``.

Atom catalogue (det-relevant):
  called(tool)
  called_any
  count(tool)
  called_with(tool, pattern)
  count_with(tool, pattern)
  consecutive_count(tool)
  arg_has(tool, pattern)
  arg_field_has(tool, field, pattern)
  arg_length_exceeds(tool, field, max_chars)
  arg_numeric(tool, field)
  arg_paths_within(tool, *prefixes)
  token_count(scope)
  delegation_depth
  ctx(key, value)
  ctx_matches(key, pattern)
  llm_said(pattern)
  response_words / response_chars
  segment(value)
  time_since(predicate_key)
  now
  perm(p) (static, from agents)
  flow(src, dst)
  contains(field)
"""

from __future__ import annotations

from contragent.formulas.formula import Atom, Var
from contragent.models import Event
from contragent.tracer.grounding import (
    GroundingState,
    collect_content_atoms,
    ground_event,
)


def _ground(events: list[Event], formulas=None):
    """Ground a sequence of events into per-step valuations."""
    state = GroundingState()
    content_atoms = collect_content_atoms(formulas) if formulas else None
    return [ground_event(ev, i, state, content_atoms) for i, ev in enumerate(events)]


def _tool(tool: str, args: dict | None = None, ts: float = 1.0, agent: str = "a") -> Event:
    return Event(event_type="tool_call", tool=tool, agent=agent, ts=ts, args=args or {})


# ── called / called_any / count ──────────────────────────────────────


def test_called_atom_fires_for_invoked_tool():
    [v] = _ground([_tool("read_file")])
    assert v.get("called(read_file)") is True
    assert v.get("called_any()") is True


def test_count_atom_accumulates_per_tool():
    vs = _ground([_tool("send_email"), _tool("send_email"), _tool("send_email")])
    assert [v["count(send_email)"] for v in vs] == [1, 2, 3]


# ── called_with / count_with ────────────────────────────────────────


def test_called_with_matches_args_regex():
    formulas = [Atom("called_with", "send_email", "spam")]
    vs = _ground(
        [
            _tool("send_email", {"to": "spam@evil.com"}),
            _tool("send_email", {"to": "ok@example.com"}),
        ],
        formulas,
    )
    assert vs[0].get("called_with(send_email, spam)") is True
    assert vs[1].get("called_with(send_email, spam)") is False


def test_count_with_accumulates_pattern_matches():
    formulas = [Var("count_with", "send_email", "spam")]
    vs = _ground(
        [
            _tool("send_email", {"to": "spam@evil.com"}),
            _tool("send_email", {"to": "ok@example.com"}),
            _tool("send_email", {"to": "spam2@evil.com"}),
        ],
        formulas,
    )
    assert [v["count_with(send_email, spam)"] for v in vs] == [1, 1, 2]


# ── consecutive_count ───────────────────────────────────────────────


def test_consecutive_count_resets_on_different_tool():
    vs = _ground(
        [_tool("poll"), _tool("poll"), _tool("done"), _tool("poll")],
    )
    assert vs[0]["consecutive_count(poll)"] == 1
    assert vs[1]["consecutive_count(poll)"] == 2
    assert vs[3]["consecutive_count(poll)"] == 1


# ── arg_has ──────────────────────────────────────────────────────────


def test_arg_has_matches_serialized_args():
    formulas = [Atom("arg_has", "execute_sql", r"DROP")]
    [v] = _ground([_tool("execute_sql", {"query": "DROP TABLE"})], formulas)
    assert v.get("arg_has(execute_sql, DROP)") is True


# ── arg_field_has ────────────────────────────────────────────────────


def test_arg_field_has_matches_named_field():
    formulas = [Atom("arg_field_has", "post", "channel", r"^#prod-")]
    [v] = _ground([_tool("post", {"channel": "#prod-alerts"})], formulas)
    assert v.get("arg_field_has(post, channel, ^#prod-)") is True


def test_arg_field_has_misses_other_field():
    formulas = [Atom("arg_field_has", "post", "channel", r"^#prod-")]
    [v] = _ground([_tool("post", {"channel": "#dev-only"})], formulas)
    assert v.get("arg_field_has(post, channel, ^#prod-)") is False


# ── arg_length_exceeds ───────────────────────────────────────────────


def test_arg_length_exceeds_long_field():
    formulas = [Atom("arg_length_exceeds", "post", "body", "10")]
    long = _tool("post", {"body": "x" * 50})
    short = _tool("post", {"body": "ok"})
    vs = _ground([long, short], formulas)
    assert vs[0].get("arg_length_exceeds(post, body, 10)") is True
    assert vs[1].get("arg_length_exceeds(post, body, 10)") is False


# ── arg_numeric ──────────────────────────────────────────────────────


def test_arg_numeric_extracts_integer_field():
    formulas = [Var("arg_numeric", "set_temp", "value")]
    [v] = _ground([_tool("set_temp", {"value": 42})], formulas)
    assert v.get("arg_numeric(set_temp, value)") == 42


def test_arg_numeric_extracts_cli_flag():
    formulas = [Var("arg_numeric", "bash", "rate")]
    [v] = _ground([_tool("bash", {"command": "send --rate 10 --batch 5"})], formulas)
    assert v.get("arg_numeric(bash, rate)") == 10


# ── arg_paths_within ─────────────────────────────────────────────────


def test_arg_paths_within_inside_prefix():
    formulas = [Atom("arg_paths_within", "write_file", "/tmp/")]
    [v] = _ground([_tool("write_file", {"path": "/tmp/output"})], formulas)
    assert v.get("arg_paths_within(write_file, /tmp/)") is True


def test_arg_paths_within_outside_prefix():
    formulas = [Atom("arg_paths_within", "write_file", "/tmp/")]
    [v] = _ground([_tool("write_file", {"path": "/etc/passwd"})], formulas)
    assert v.get("arg_paths_within(write_file, /tmp/)") is False


# ── token_count ──────────────────────────────────────────────────────


def test_token_count_accumulates_total():
    formulas = [Var("token_count", "total")]
    e1 = _tool("ask_llm", {"tokens": 100})
    e2 = _tool("ask_llm", {"tokens": 50})
    vs = _ground([e1, e2], formulas)
    assert vs[0]["token_count(total)"] == 100
    assert vs[1]["token_count(total)"] == 150


# ── delegation_depth ─────────────────────────────────────────────────


def test_delegation_depth_increments_on_message():
    e1 = Event(event_type="message", agent="a", ts=1.0, to="b")
    e2 = Event(event_type="message", agent="b", ts=2.0, to="c")
    e3 = Event(event_type="message", agent="c", ts=3.0, to="d")
    vs = _ground([e1, e2, e3])
    assert [v["delegation_depth()"] for v in vs] == [1, 2, 3]


# ── ctx / ctx_matches ────────────────────────────────────────────────


def test_ctx_atom_emitted_after_context_update():
    state = GroundingState()
    upd = Event(event_type="context_update", agent="a", ts=1.0, args={"caller_id": "alice"})
    tool = _tool("wire_transfer", ts=2.0)
    v0 = ground_event(upd, 0, state)
    v1 = ground_event(tool, 1, state)
    assert v0.get("ctx(caller_id, alice)") is True
    assert v1.get("ctx(caller_id, alice)") is True


def test_ctx_matches_evaluates_regex_against_current_ctx():
    formulas = [Atom("ctx_matches", "approval.role", r"senior_eng")]
    state = GroundingState()
    content_atoms = collect_content_atoms(formulas)
    upd = Event(
        event_type="context_update",
        agent="a",
        ts=1.0,
        args={"approval.role": "senior_eng"},
    )
    tool = _tool("refund", ts=2.0)
    ground_event(upd, 0, state, content_atoms)
    v = ground_event(tool, 1, state, content_atoms)
    assert v.get("ctx_matches(approval.role, senior_eng)") is True


# ── llm_said / response_words / response_chars / segment ─────────────


def test_llm_said_matches_regex_against_response():
    atom = Atom("llm_said", r"\bsecret\b")
    e = Event(event_type="llm_response", agent="a", ts=1.0, content="the secret is here")
    [v] = _ground([e], [atom])
    assert v.get(atom.key()) is True


def test_response_words_and_chars_emitted():
    e = Event(
        event_type="llm_response",
        agent="a",
        ts=1.0,
        content="five word response is fine",
    )
    [v] = _ground([e])
    assert v.get("response_words") == 5
    assert v.get("response_chars") == 26


def test_segment_atom_for_thinking_tag():
    e = Event(
        event_type="llm_response",
        agent="a",
        ts=1.0,
        content="…",
        args={"segment": "thinking"},
    )
    [v] = _ground([e])
    assert v.get("segment(thinking)") is True


# ── time_since ───────────────────────────────────────────────────────
# ``time_since`` keys live under ``Var.key()`` (which routes through
# pred_key and so escapes parens/commas/spaces in the predicate name).
# Look up via ``var.key()`` instead of hand-formatting strings.


def test_time_since_zero_when_predicate_just_fired():
    var = Var("time_since", "ctx(approval, granted)")
    state = GroundingState()
    content_atoms = collect_content_atoms([var])
    upd = Event(event_type="context_update", agent="a", ts=1.0, args={"approval": "granted"})
    v = ground_event(upd, 0, state, content_atoms)
    # ``last_ts`` updated to ts=1.0 in this event; delta = 0.
    assert v.get(var.key()) == 0


def test_time_since_advances_with_clock():
    var = Var("time_since", "ctx(approval, granted)")
    state = GroundingState()
    content_atoms = collect_content_atoms([var])
    upd = Event(event_type="context_update", agent="a", ts=1.0, args={"approval": "granted"})
    later = _tool("act", ts=10.0)
    ground_event(upd, 0, state, content_atoms)
    v = ground_event(later, 1, state, content_atoms)
    assert v.get(var.key()) == 9


def test_time_since_sentinel_when_predicate_never_fired():
    var = Var("time_since", "ctx(approval, granted)")
    [v] = _ground([_tool("act", ts=1.0)], [var])
    assert v.get(var.key()) == 1e18


# ── now ─────────────────────────────────────────────────────────────


def test_now_atom_tracks_event_clock():
    [v] = _ground([_tool("act", ts=42.0)])
    assert v.get("now") == 42.0


# ── perm (from agents) ──────────────────────────────────────────────


def test_perm_atom_from_agent_permissions():
    from contragent.models import Agent

    agents = {"a": Agent(id="a", permissions=["admin"])}
    state = GroundingState()
    e = _tool("delete_account", ts=1.0)
    v = ground_event(e, 0, state, agents=agents)
    assert v.get("perm(admin)") is True


# ── flow / contains ─────────────────────────────────────────────────


def test_flow_atom_from_data_read_after_write():
    e1 = Event(event_type="data_write", agent="writer", ts=1.0, key="doc", contains=["pii"])
    e2 = Event(event_type="data_read", agent="reader", ts=2.0, key="doc")
    vs = _ground([e1, e2])
    assert vs[0].get("contains(pii)") is True
    assert vs[1].get("flow(writer, reader)") is True


def test_contains_atom_propagates_forward():
    e1 = Event(event_type="data_write", agent="bot", ts=1.0, key="customer", contains=["pii"])
    e2 = _tool("ping", ts=2.0)
    vs = _ground([e1, e2])
    # ``contains(pii)`` should still be True at the next event due to
    # forward propagation in the grounding state.
    assert vs[1].get("contains(pii)") is True
