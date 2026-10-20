#!/usr/bin/env python3
"""Analyze admin-auth-before-transfer visibility in pinned SOPBench bank traces.

Reads the bank output JSONs from the existing checkout, or fetches them into a
temporary directory with --download. Writes a stacked SVG and JSON summary next
to this script. A trace counts as scoped only when its task constraints include
the authenticated_admin_password gate.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_transfer_no_call_reasons import DEFAULT_INPUT, HERE, UPSTREAM_COMMIT, download_records, read_records


def contains(value: Any, target: str) -> bool:
    if value == target:
        return True
    if isinstance(value, dict):
        return any(contains(v, target) for v in value.values())
    if isinstance(value, list):
        return any(contains(v, target) for v in value)
    return False


def conversation(record: dict[str, Any]) -> list[dict[str, Any]]:
    interactions = record.get("interactions") or []
    return (interactions[0].get("interaction") or []) if interactions else []


def events(record: dict[str, Any]) -> list[tuple[int, str, str | None]]:
    """Return ordered tool events (message index, name, result text)."""
    result_by_id: dict[str, str] = {}
    for i, message in enumerate(conversation(record)):
        if message.get("tool_call_id"):
            result_by_id[message["tool_call_id"]] = str(message.get("content", "")).strip()
    out: list[tuple[int, str, str | None]] = []
    for i, message in enumerate(conversation(record)):
        for call in message.get("tool_calls") or []:
            name = (call.get("function") or {}).get("name")
            if name:
                out.append((i, name, result_by_id.get(call.get("id"))))
    return out


def scenario_key(task: dict[str, Any]) -> str:
    """Stable task-level key across model/config repetitions."""
    selected = {
        "constraints": task.get("constraints"),
        "user_known": task.get("user_known"),
        "action_should_succeed": task.get("action_should_succeed"),
        "initial_database": task.get("initial_database"),
    }
    return json.dumps(selected, sort_keys=True, separators=(",", ":"), default=str)


def classify(record: dict[str, Any]) -> str:
    ev = events(record)
    transfers = [e for e in ev if e[1] == "transfer_funds"]
    auths = [e for e in ev if e[1] == "authenticate_admin_password"]
    if not transfers:
        return "No transfer attempted"
    first_transfer = min(e[0] for e in transfers)
    prior_auths = [e for e in auths if e[0] < first_transfer]
    if prior_auths:
        # Require the most recent prior auth result to be explicit.
        result = prior_auths[-1][2]
        if result is not None and result.lower() == "true":
            return "Successful admin auth before transfer"
        if result is not None and result.lower() == "false":
            return "Failed admin auth before transfer"
        return "Admin auth before transfer; result unclear"
    if auths:
        return "Admin auth only after transfer"
    return "No admin-auth call"


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    scoped = [
        r for r in records
        if (r.get("task") or {}).get("user_goal") == "transfer_funds"
        and contains((r.get("task") or {}).get("constraints", []), "authenticated_admin_password")
    ]
    pattern_counts: dict[str, Counter[str]] = defaultdict(Counter)
    all_counts: Counter[str] = Counter()
    task_patterns: dict[str, set[str]] = defaultdict(set)
    task_success: dict[str, bool] = {}
    task_pattern_counts: dict[str, Counter[str]] = defaultdict(Counter)
    successful_transfer_followups: Counter[str] = Counter()
    for record in scoped:
        task = record.get("task") or {}
        pattern = classify(record)
        key = scenario_key(task)
        should = "Permitted" if task.get("action_should_succeed") else "Forbidden"
        pattern_counts[should][pattern] += 1
        all_counts[pattern] += 1
        task_patterns[should].add(key)
        task_success[key] = bool(task.get("action_should_succeed"))
        task_pattern_counts[key][pattern] += 1
        ev = events(record)
        successful_transfers = [e for e in ev if e[1] == "transfer_funds" and e[2] and e[2].lower() == "true"]
        if successful_transfers:
            first_success = successful_transfers[0]
            later_tools = [e for e in ev if e[0] > first_success[0]]
            if not later_tools:
                successful_transfer_followups["No later tool call in trace"] += 1
            elif later_tools[0][1] == "exit_conversation":
                successful_transfer_followups["Exit conversation is next tool"] += 1
            else:
                successful_transfer_followups["Another tool is next"] += 1

    # Keep zero-valued patterns explicit so absence of counterexamples is
    # visible in the machine-readable result instead of being omitted.
    for pattern in (
        "Successful admin auth before transfer",
        "Failed admin auth before transfer",
        "Admin auth before transfer; result unclear",
        "Admin auth only after transfer",
        "No admin-auth call",
        "No transfer attempted",
    ):
        all_counts.setdefault(pattern, 0)

    task_level: dict[str, Counter[str]] = {"Permitted": Counter(), "Forbidden": Counter()}
    for key, counts in task_pattern_counts.items():
        label = "Permitted" if task_success[key] else "Forbidden"
        # For each task scenario, report the most common observed pattern;
        # the set of patterns below preserves variation within the scenario.
        task_level[label][counts.most_common(1)[0][0]] += 1

    varied = {
        "scenarios_with_multiple_patterns": sum(len(set(c)) > 1 for c in task_pattern_counts.values()),
        "scenarios_with_successful_auth_before_transfer": sum(
            c.get("Successful admin auth before transfer", 0) > 0 for c in task_pattern_counts.values()
        ),
        "scenarios_with_failed_auth_before_transfer": sum(
            c.get("Failed admin auth before transfer", 0) > 0 for c in task_pattern_counts.values()
        ),
        "scenarios_with_no_auth_call": sum(c.get("No admin-auth call", 0) > 0 for c in task_pattern_counts.values()),
    }
    return {
        "source": {"repository": "https://github.com/Leezekun/SOPBench", "commit": UPSTREAM_COMMIT, "domain": "bank"},
        "scope": "transfer_funds traces whose task constraints include authenticated_admin_password",
        "scoped_run_records": len(scoped),
        "unique_task_scenarios": {k: len(v) for k, v in task_patterns.items()},
        "run_patterns_by_task_permission": {k: dict(v) for k, v in pattern_counts.items()},
        "run_patterns_all": dict(all_counts),
        "after_successful_transfer": dict(successful_transfer_followups),
        "unique_scenario_majority_pattern": {k: dict(v) for k, v in task_level.items()},
        "scenario_variation": varied,
        "limitations": [
            "Repeated configurations are not independent runs; unique scenarios are separately counted.",
            "A true tool return is the logged result of authenticate_admin_password; the trace does not prove the model's internal belief.",
            "Task permission (action_should_succeed) can be false because of another blocker, so it is not an auth-validity label.",
            "Task gate context is derived from task.constraints; mining from traces alone would need this context encoded or inferred.",
        ],
    }


def plot(summary: dict[str, Any], path: Path) -> None:
    patterns = [
        "Successful admin auth before transfer",
        "Failed admin auth before transfer",
        "Admin auth before transfer; result unclear",
        "Admin auth only after transfer",
        "No admin-auth call",
        "No transfer attempted",
    ]
    colors = ["#327a5b", "#c85a4a", "#a277b5", "#e2a94b", "#93a7b1", "#d9dfe2"]
    groups = ["Permitted", "Forbidden"]
    counts = summary["run_patterns_by_task_permission"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 6.5), gridspec_kw={"width_ratios": [1.55, 1]})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y_positions = [1, 0]
    left = [0, 0]
    for pattern, color in zip(patterns, colors):
        vals = [counts.get(group, {}).get(pattern, 0) for group in groups]
        ax.barh(y_positions, vals, left=left, color=color, height=0.52, label=pattern, edgecolor="white", linewidth=1)
        for y, l, v in zip(y_positions, left, vals):
            if v >= 8:
                ax.text(l + v / 2, y, str(v), ha="center", va="center", color="white" if color not in ("#d9dfe2",) else "#34434a", fontsize=9, fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]
    ax.set_yticks(y_positions, ["Permitted tasks", "Forbidden tasks"])
    ax.set_xlabel("Run records (repeated configurations)")
    ax.set_title("Observed event order", loc="left", fontsize=13, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c8d0d3")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e1e6e8", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False, fontsize=8.2)

    unique = summary["unique_scenario_majority_pattern"]
    left = [0, 0]
    for pattern, color in zip(patterns, colors):
        vals = [unique.get(group, {}).get(pattern, 0) for group in groups]
        ax2.barh(y_positions, vals, left=left, color=color, height=0.52, edgecolor="white", linewidth=1)
        for y, l, v in zip(y_positions, left, vals):
            if v:
                ax2.text(l + v / 2, y, str(v), ha="center", va="center", color="white" if color not in ("#d9dfe2",) else "#34434a", fontsize=9, fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]
    ax2.set_yticks(y_positions, ["Permitted tasks", "Forbidden tasks"])
    ax2.set_xlabel("Unique scenarios\n(most common pattern per scenario)")
    ax2.set_title("Scenario-level summary", loc="left", fontsize=13, fontweight="bold")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.spines["bottom"].set_color("#c8d0d3")
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="x", color="#e1e6e8", zorder=0)
    ax2.set_axisbelow(True)
    total = summary["scoped_run_records"]
    fig.suptitle(f"Admin-auth ordering in bank transfer traces · {total:,} gate-scoped runs", x=0.055, y=0.97, ha="left", fontsize=16, fontweight="bold", color="#1f3037")
    fig.text(0.055, 0.915, "A task is scoped only when its SOP includes the admin-auth gate. Bars count traces; the right panel prevents repeated configs from looking like independent tasks.", ha="left", fontsize=9, color="#52656d")
    fig.text(0.055, 0.125, f"Order counterexamples: {summary['run_patterns_all']['Admin auth only after transfer']} traces show admin auth only after transfer.", ha="left", fontsize=9, color="#8b3e34", fontweight="bold")
    fig.text(0.055, 0.045, "‘Before’ means the tool call precedes the first transfer call; auth result is read from the matching tool response.", ha="left", fontsize=8.5, color="#52656d")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.28, wspace=0.42)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--svg", type=Path, default=HERE / "admin-auth-order.svg")
    parser.add_argument("--summary", type=Path, default=HERE / "admin-auth-order-summary.json")
    args = parser.parse_args()
    if args.download:
        with tempfile.TemporaryDirectory(prefix="sopbench-bank-auth-") as temp:
            records, files = download_records(Path(temp))
            result = analyze(records)
    else:
        records, files = read_records(args.input_dir)
        result = analyze(records)
    result["input_files"] = len(files)
    result["input_file_names"] = files
    args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot(result, args.svg)
    print(json.dumps({"scoped_runs": result["scoped_run_records"], "unique_scenarios": result["unique_task_scenarios"], "run_patterns": result["run_patterns_by_task_permission"], "scenario_variation": result["scenario_variation"], "svg": str(args.svg), "summary": str(args.summary)}, indent=2))


if __name__ == "__main__":
    main()
