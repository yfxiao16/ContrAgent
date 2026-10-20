"""Shared prompt, trace conversion, and evaluation helpers for SOP trace mining.

Domain-specific code should only define the trace source, gate extraction, and
the tools that form its candidate ordering rule. Keep this module unaware of
banking or healthcare task semantics.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contragent import ContrAgent, Event, Trace, contract, parse_repr


def format_training_prompt(instructions: str, examples: list[dict[str, Any]]) -> str:
    """Append the same auditable JSON training block for any domain miner."""
    return (
        instructions.rstrip()
        + "\n\nTraining examples (JSON):\n"
        + json.dumps(examples, indent=2, ensure_ascii=False)
        + "\n"
    )


def extract_formula(text: str) -> str:
    """Extract the one-line formula returned by a model."""
    cleaned = text.replace("```", "").strip()
    match = re.search(r"(?:^|\n)\s*(?:G\s*:\s*|formula\s*:\s*)(.+)", cleaned, flags=re.IGNORECASE)
    if match:
        formula = match.group(1).strip()
    else:
        formula = next((line.strip() for line in cleaned.splitlines() if "called(" in line or "G(" in line), "")
    return formula.strip().strip("`").strip('"').strip("'")


def result_succeeded(value: Any) -> bool:
    """Read the Boolean success prefix used by SOPBench tool responses."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    rendered = str(value or "").strip()
    if rendered.startswith(("(", "[")):
        rendered = rendered[1:].lstrip()
    return rendered.startswith("True") or rendered.lower() == "true"


def ordering_label(
    item: dict[str, Any],
    *,
    prerequisite_tool: str,
    goal_tool: str,
    reset_tool: str | None = None,
) -> str:
    """Label conformance to a scoped successful-prerequisite-before-goal rule."""
    if not item.get("gate_active", True):
        return "conforms"
    prerequisite_seen = False
    for event in item.get("events", []):
        tool = event.get("tool")
        succeeded = result_succeeded(event.get("succeeded"))
        if tool == reset_tool and succeeded:
            prerequisite_seen = False
        elif tool == prerequisite_tool and succeeded:
            prerequisite_seen = True
        elif tool == goal_tool and succeeded and not prerequisite_seen:
            return "violates"
    return "conforms"


def to_contragent_trace(
    item: dict[str, Any],
    *,
    agent_id: str,
    goal_tool: str,
    gate_arg: str,
) -> Trace:
    """Convert a sanitized mining trace to ContrAgent's native Trace type."""
    events = []
    for ts, raw in enumerate(item.get("events", [])):
        args: dict[str, Any] = {"succeeded": int(result_succeeded(raw.get("succeeded")))}
        if raw.get("tool") == goal_tool:
            args[gate_arg] = int(bool(item.get("gate_active", True)))
        events.append(
            Event(
                ts=ts,
                agent=agent_id,
                event_type="tool_call",
                tool=raw.get("tool", ""),
                args=args,
            )
        )
    return Trace(events=events, metadata={"gate_active": bool(item.get("gate_active", True))})


def evaluate_candidate(
    formula_text: str,
    items: list[dict[str, Any]],
    *,
    agent_id: str,
    goal_tool: str,
    prerequisite_tool: str,
    gate_arg: str,
    reset_tool: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Evaluate a candidate formula and return per-trace reference comparisons."""
    try:
        formula = parse_repr(formula_text)
    except Exception as exc:
        return [], f"Formula parse failed: {type(exc).__name__}: {exc}"
    guard = ContrAgent(
        agent_id=agent_id,
        contracts=[contract("candidate trace-mined ordering rule").guarantees(formula)],
    )
    evaluated = []
    for item in items:
        verdict = guard.evaluate_trace(
            to_contragent_trace(item, agent_id=agent_id, goal_tool=goal_tool, gate_arg=gate_arg),
            include_liveness=True,
        )
        predicted = "violates" if verdict["verdict"] == "FAIL" else "conforms"
        expected = ordering_label(
            item,
            prerequisite_tool=prerequisite_tool,
            goal_tool=goal_tool,
            reset_tool=reset_tool,
        )
        evaluated.append(
            {
                "trace_id": item["trace_id"],
                "expected": expected,
                "predicted": predicted,
                "verdict": verdict["verdict"],
                "correct": predicted == expected,
                "gate_active": bool(item.get("gate_active", True)),
                "violations": verdict.get("violations", []),
            }
        )
    return evaluated, None


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Confusion counts with a consistent meaning across domain experiments."""
    tp = sum(r["expected"] == "violates" and r["predicted"] == "violates" for r in rows)
    tn = sum(r["expected"] == "conforms" and r["predicted"] == "conforms" for r in rows)
    fp = sum(r["expected"] == "conforms" and r["predicted"] == "violates" for r in rows)
    fn = sum(r["expected"] == "violates" and r["predicted"] == "conforms" for r in rows)
    return {
        "n": len(rows),
        "correct": tp + tn,
        "accuracy": (tp + tn) / len(rows) if rows else None,
        "true_violations": tp,
        "false_rejections": fp,
        "missed_violations": fn,
        "true_conforms": tn,
    }
