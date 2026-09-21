#!/usr/bin/env python3
"""R-Judge records -> ContrAgent NATIVE trace JSON.

R-Judge (https://github.com/Lordog/R-Judge) is a *detection* benchmark: each
record is a multi-turn agent transcript labelled ``0`` (safe) / ``1`` (unsafe).
We emit one native-trace JSON per record under ``traces/``, named
``safe_rjudge_<...>.json`` / ``unsafe_rjudge_<...>.json`` so ``contragent eval``
scores them by filename prefix.

THE PRINCIPLE — this file is a PURE, FAITHFUL FORMAT TRANSLATOR.

It does exactly one thing: read each turn of an R-Judge transcript and emit the
equivalent ContrAgent event, preserving the REAL tool name, the REAL args, and
the REAL text. It contains:

  * NO semantic judgment. It never classifies a tool into a capability bucket,
    never invents synthetic class tools (no ``external_send`` / ``agent_effect``
    / ``untrusted_msg_read`` ...), never scans for injection markers, never
    reads the ``label`` field.
  * Only format parsing. ``GmailSendEmail: {"to": ...}`` becomes
    ``tool="GmailSendEmail", args={"to": ...}`` — that is faithful translation
    of the on-disk action syntax, not a semantic decision.

ALL policy lives in the hand-authored LTL in ``contracts.yaml``, where the tool
groups are declared as explicit, reviewer-visible lists of these REAL tool
names.

Native trace schema emitted (loaded by contragent.eval_runner via
Trace.from_dict — see eval_runner.discover_cases: a dict with an ``events`` key
and no ``resourceSpans`` is parsed as native):

    {"metadata": {... id, category, label, attack_type ...},
     "events": [
        {"ts", "agent", "type": "tool_call", "tool": <real>, "args": <real>,
         "content": <real action/observation text>},
        {"ts", "agent", "type": "message",   "content": <real user/env text>},
        {"ts", "agent", "type": "llm_response", "content": <real free text>},
     ]}

NOTE ON GROUNDING (why the events are shaped this way, NOT a policy choice):
``contragent eval`` grounds with ``ground(trace)`` and passes NO content_atoms
(see contragent/eval_runner.py::_eval_contract_on_trace). Content-matching
predicates (arg_has / output_has / llm_said) therefore do not fire in the eval
harness — only STRUCTURAL predicates (called / called_any / count /
consecutive_count) are observable. The converter still records real args/content
faithfully (so the traces are honest and re-usable with content_atoms), but the
shipped contracts key on the real tool *names*, grouped in the YAML.

Action formats handled (all seen in the data) — pure syntax, no semantics:
    dict actions:        {"name": ..., "args": ...}
    ``Name: {json}``  /  ``Name: [json]``  /  ``Name{json}``
    ``Name\nAction Input: {json}``
    AutoGPT envelope:    ``"command": {"name": "X", "args": {...}}``
    AutoGPT shell:       ``"command": "<shell string>"``  -> tool TerminalExecute
    fenced bash block:   ```` ```bash ... ``` ````        -> tool TerminalExecute
    bare ``Name()``
    anything else        -> free text (no tool call) -> llm_response event

Usage:  python convert.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "traces"

CATEGORIES = ["Application", "Finance", "IoT", "Program", "Web"]

_TOOLNAME = r"[A-Za-z][A-Za-z0-9_]*"


# ---------------------------------------------------------------------------
# Pure format parsing:  action -> (tool_name | None, args_dict, raw_text)
#
# ``None`` tool name => the action is plain free text (no tool call). NOTHING
# here inspects what the tool *does*; it only recognizes the on-disk syntax of a
# tool invocation and extracts the literal name + literal args.
# ---------------------------------------------------------------------------


def parse_action(action) -> tuple[str | None, dict, str]:
    if isinstance(action, dict):
        name = action.get("name") or action.get("tool") or action.get("action")
        args = action.get("arguments") or action.get("args") or action.get("input") or {}
        if not isinstance(args, dict):
            args = {"_raw": args}
        return (str(name) if name else None), args, json.dumps(action)[:6000]
    if not isinstance(action, str):
        return None, {}, ""
    s = action.strip()
    raw = s[:6000]
    if not s:
        return None, {}, raw

    # Fenced shell block  -> the tool the agent invoked is the terminal.
    if re.search(r"```\s*(?:bash|sh|shell|zsh)", s) or re.match(r"^bash\b", s, re.I):
        m = re.search(r"```(?:bash|sh|shell|zsh)?\s*(.*?)```", s, re.S)
        cmd = m.group(1).strip() if m else re.sub(r"^bash\b", "", s, flags=re.I).strip()
        return "TerminalExecute", {"command": cmd}, raw

    # AutoGPT command envelope:  {"command": {"name": "X", "args": {...}}}
    # (often emitted bare / unbalanced in the data, so a regex extracts it.)
    m = re.search(r'"command"\s*:\s*\{[^}]*?"name"\s*:\s*"([a-zA-Z_]+)"', s)
    if m:
        name = m.group(1)
        am = re.search(r'"args"\s*:\s*(\{.*)$', s, re.S)
        return name, (_parse_args(am.group(1)) if am else {}), raw
    # AutoGPT shell envelope:  {"command": "<shell string>"}
    m = re.search(r'"command"\s*:\s*"([^"]+)"', s)
    if m:
        return "TerminalExecute", {"command": m.group(1)}, raw

    # ``Name\nAction Input: {json}``  (ReAct format)
    m = re.match(rf"^({_TOOLNAME})\s*\n+\s*Action Input:\s*(.*)$", s, re.S)
    if m:
        return m.group(1), _parse_args(m.group(2)), raw

    # ``Name: {json}`` / ``Name: [json]`` — only a tool call if what follows the
    # colon is a JSON object/array. ``Woman: I love you`` (dialog prose) has
    # plain text after the colon and is therefore NOT a tool call. This is a
    # syntactic test (is the payload JSON?), not a semantic one.
    m = re.match(rf"^({_TOOLNAME})\s*:\s*([\{{\[].*)$", s, re.S)
    if m:
        return m.group(1), _parse_args(m.group(2)), raw
    # ``Name{json}`` with no colon.
    m = re.match(rf"^({_TOOLNAME})\s*(\{{.*)$", s, re.S)
    if m:
        return m.group(1), _parse_args(m.group(2)), raw
    # ``Name()`` — empty-arg call.
    m = re.match(rf"^({_TOOLNAME})\s*\(\s*\)\s*$", s)
    if m:
        return m.group(1), {}, raw

    # ``Name(<args>)`` — a single identifier whose whole payload is one
    # parenthesised argument list, e.g. ``Delete('task_id': 3)`` or
    # ``PostBlog(id = 20231018183200)``. This is the bare function-call syntax
    # some R-Judge agents emit instead of ``Name: {json}``. It is recognised the
    # same syntactic way as the other forms (leading identifier + delimited
    # payload); the payload is rarely valid JSON here, so it is preserved raw.
    # The anchored ^...$ keeps prose ("I will now stop (for today)") from
    # matching — only a lone identifier directly followed by ``(`` qualifies.
    m = re.match(rf"^({_TOOLNAME})\s*\((.+)\)\s*$", s, re.S)
    if m:
        return m.group(1), _parse_args(m.group(2)), raw

    # Bare verb-prefixed AutoGPT actions occasionally appear unwrapped, e.g.
    # ``write_to_file: {...}`` / ``read_file: {...}``. Same JSON-payload test as
    # above already covers the colon+JSON case; this only adds the names so the
    # leading-identifier rule is applied. (Redundant with the colon rule, kept
    # for clarity of intent.)
    return None, {}, raw


def _parse_args(rest: str) -> dict:
    rest = rest.strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(rest)
            return obj if isinstance(obj, dict) else {"_raw": obj}
        except Exception:
            continue
    return {"_raw": rest}


# ---------------------------------------------------------------------------
# Record -> native events
# ---------------------------------------------------------------------------


def record_to_events(rec: dict) -> list[dict]:
    events: list[dict] = []
    ts = 1

    for turn in rec.get("contents") or []:
        for m in turn if isinstance(turn, list) else [turn]:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "").lower()

            if role in ("user", "environment"):
                # User instruction or environment/tool observation — carry the
                # REAL text verbatim as a message event. (No marker detection,
                # no PII scan; just the text.)
                content = m.get("content")
                if content is None:
                    continue
                events.append(
                    {
                        "ts": ts,
                        "agent": "user" if role == "user" else "env",
                        "type": "message",
                        "to": "agent",
                        "content": str(content)[:6000],
                    }
                )
                ts += 1

            elif role == "agent":
                tool, args, raw = parse_action(m.get("action"))
                if tool is not None:
                    # Faithful tool call: real name, real args, real action text.
                    events.append(
                        {
                            "ts": ts,
                            "agent": "agent",
                            "type": "tool_call",
                            "tool": tool,
                            "args": args,
                            "content": raw,
                        }
                    )
                    ts += 1
                else:
                    # Free-text agent turn (a Final Answer / refusal / dialog).
                    # Emit the real text as an llm_response; no tool call.
                    text = raw or (str(m.get("thought")) if m.get("thought") else "")
                    if text:
                        events.append(
                            {
                                "ts": ts,
                                "agent": "agent",
                                "type": "llm_response",
                                "content": text[:6000],
                            }
                        )
                        ts += 1

    return events


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob("*.json"):
        p.unlink()

    total = safe = unsafe = 0
    per_cat: dict[str, list[int]] = {}
    for cat in CATEGORIES:
        cdir = DATA / cat
        if not cdir.is_dir():
            continue
        for src in sorted(cdir.glob("*.json")):
            for rec in json.load(open(src)):
                rid = rec.get("id")
                label = int(rec.get("label", 0))
                prefix = "unsafe" if label == 1 else "safe"
                events = record_to_events(rec)
                trace = {
                    "metadata": {
                        "id": rid,
                        "category": cat,
                        "source_file": src.name,
                        "label": label,
                        "attack_type": rec.get("attack_type"),
                        "scenario": rec.get("scenario"),
                    },
                    "events": events,
                }
                name = f"{prefix}_rjudge_{cat}_{src.stem}_{rid}.json"
                (OUT / name).write_text(json.dumps(trace))
                total += 1
                if label == 1:
                    unsafe += 1
                else:
                    safe += 1
                per_cat.setdefault(cat, [0, 0])
                per_cat[cat][label] += 1

    print(f"wrote {total} native traces ({safe} safe / {unsafe} unsafe) -> {OUT}")
    for cat, (s, u) in sorted(per_cat.items()):
        print(f"  {cat}: {s} safe / {u} unsafe")


if __name__ == "__main__":
    main()
