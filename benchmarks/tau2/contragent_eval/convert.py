"""tau2-bench trace -> ContrAgent native trace converter.

Faithful, *thin* converter from a tau2-bench simulation transcript (one
entry of the ``simulations`` list in a
``data/tau2/results/final/<model>_<domain>_..._4trials.json`` file) into
ContrAgent's native trace JSON::

    {"metadata": {...}, "events": [{"ts","agent","type","tool","args","content"}]}

Design principle: **REAL tool names / args only. No label peeking, no
semantic tagging.**

What is emitted (all structurally observable in the transcript):

  * One ``tool_call`` event per tool call in every assistant / user
    message (tau2 telecom has user-side device-diagnostic tool calls;
    we keep the real ``requestor`` as the event ``agent``). The event
    carries the REAL ``tool`` name and the REAL ``args`` dict the model
    emitted. Nothing is inferred.

  * The ``same_turn_text_and_tool_call`` protocol flag. tau2's policy
    forbids an assistant message that contains BOTH user-facing text and
    a tool call. This is a *purely structural* property of one message
    (``role == "assistant" and tool_calls and content.strip()``), so it
    is honest to surface it. We bracket the offending tool_call events
    with ``context_update`` events that set the ctx flag to ``"yes"``
    just before and reset it to ``"no"`` just after, so the
    ``G(!ctx(same_turn_text_and_tool_call, yes))`` contract fires at
    exactly the violating timesteps and nowhere else.

What is deliberately NOT emitted: the policy-derived ctx facts
(``user_authenticated``, ``target_order_status``, ``order_owner_match``,
``account_status``, ``target_line_status`` ...). Deriving those requires
re-implementing the domain policy against the environment DB, which is
exactly the semantic tagging this converter refuses to do. Contracts
that depend on those ctx atoms are therefore *not honestly evaluable*
from the transcript alone and are reported separately as "skipped
(needs derived ctx)" by ``eval_proc.py``.

The tau2 ``reward_info.reward`` (1.0 == task passed) is copied into
trace ``metadata`` so the eval can compute the blind-spot statistic
(tau2-passing sims that still fire a procedure contract).
"""

from __future__ import annotations

from typing import Any


# Logical-clock counter is per-trace; ts only needs to be monotonic.
def tau2_sim_to_trace(sim: dict[str, Any], *, model: str, domain: str) -> dict[str, Any]:
    """Convert one tau2 simulation dict to a ContrAgent native trace dict.

    Args:
        sim: One element of the ``simulations`` list in a tau2 results file.
        model: Model id (for trace metadata).
        domain: Domain name retail|airline|telecom (for trace metadata).

    Returns:
        A dict ``{"metadata": ..., "events": [...]}`` ready for
        ``contragent.models.trace.Trace.from_dict``.
    """
    events: list[dict[str, Any]] = []
    ts = 0

    for msg in sim.get("messages", []):
        role = msg.get("role")
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue

        content = (msg.get("content") or "").strip()
        # Same-turn protocol flag is only meaningful for assistant
        # messages (the agent is the party bound by the policy). The
        # user simulator emitting text alongside a device-diagnostic
        # call is not an agent-policy violation.
        same_turn = bool(content) and role == "assistant"

        if same_turn:
            ts += 1
            events.append(
                {
                    "ts": ts,
                    "agent": "assistant",
                    "type": "context_update",
                    "args": {"same_turn_text_and_tool_call": "yes"},
                }
            )

        for tc in tool_calls:
            name = tc.get("name")
            if not name:
                continue
            # ``requestor`` is the real party that issued the call
            # (assistant or, for telecom device ops, user). Keep it as
            # the event agent so it is never silently re-attributed.
            requestor = tc.get("requestor") or role or "assistant"
            args = tc.get("arguments")
            if not isinstance(args, dict):
                args = {} if args is None else {"_value": args}
            ts += 1
            events.append(
                {
                    "ts": ts,
                    "agent": requestor,
                    "type": "tool_call",
                    "tool": name,
                    "args": args,
                }
            )

        if same_turn:
            ts += 1
            events.append(
                {
                    "ts": ts,
                    "agent": "assistant",
                    "type": "context_update",
                    "args": {"same_turn_text_and_tool_call": "no"},
                }
            )

    reward_info = sim.get("reward_info") or {}
    reward = reward_info.get("reward")
    metadata = {
        "source": "tau2-bench",
        "model": model,
        "domain": domain,
        "sim_id": sim.get("id"),
        "task_id": sim.get("task_id"),
        "trial": sim.get("trial"),
        "tau2_reward": reward,
        "tau2_pass": (reward is not None and reward >= 1.0),
        "termination_reason": sim.get("termination_reason"),
    }
    return {"metadata": metadata, "events": events}


def iter_traces_from_results(results: dict[str, Any], *, model: str, domain: str):
    """Yield native-trace dicts for every simulation in a tau2 results file.

    Args:
        results: The parsed JSON of a tau2 ``..._4trials.json`` file
            (keys: ``timestamp``, ``info``, ``tasks``, ``simulations``).
        model: Model id for metadata.
        domain: Domain name for metadata.
    """
    for sim in results.get("simulations", []):
        yield tau2_sim_to_trace(sim, model=model, domain=domain)
