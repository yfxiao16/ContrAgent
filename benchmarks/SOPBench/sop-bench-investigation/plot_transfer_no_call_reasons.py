#!/usr/bin/env python3
"""Summarize bank SOPBench transfer-goal records with no transfer call.

By default, reads SOPBench output JSON files from
``benchmarks/SOPBench/output/bank``. Add ``--download`` to fetch the pinned
upstream bank outputs into a temporary directory instead. Writes an SVG and a
machine-readable JSON summary next to this script (or to the paths supplied).

The task's ``action_should_succeed`` and evaluator fields are used only for
analysis and chart labeling; no evaluator is run and no model is called.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
DEFAULT_INPUT = REPO_ROOT / "benchmarks" / "SOPBench" / "output" / "bank"
UPSTREAM_COMMIT = "8790ed7406acbd22270f90bd1c4038de3fc5db8f"
UPSTREAM_API = (
    "https://api.github.com/repos/Leezekun/SOPBench/contents/output/bank"
    f"?ref={UPSTREAM_COMMIT}"
)
RAW_BASE = f"https://raw.githubusercontent.com/Leezekun/SOPBench/{UPSTREAM_COMMIT}/"

# A false evaluator flag is a signal, not necessarily a unique root cause.
# These are plotted separately because several may be false for one record.
FAILURE_SIGNALS = [
    ("Procedure graph not satisfied", "dirgraph_satisfied"),
    ("SOP constraint violated", "constraint_not_violated"),
    ("Database result mismatch", "database_match"),
    ("Tool-call error", "no_tool_call_error"),
]


def has_transfer_call(record: dict[str, Any]) -> bool:
    interactions = record.get("interactions") or []
    if not interactions:
        return False
    conversation = interactions[0].get("interaction") or []
    return any(
        (call.get("function") or {}).get("name") == "transfer_funds"
        for message in conversation
        for call in (message.get("tool_calls") or [])
    )


def evaluator(record: dict[str, Any]) -> dict[str, Any]:
    evaluations = record.get("evaluations") or []
    if not evaluations:
        return {}
    goal = (record.get("task") or {}).get("user_goal")
    return next((item for item in evaluations if item.get("user_goal") == goal), evaluations[0])


def read_records(input_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON output files found in {input_dir}")
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as file:
            records.extend(json.load(file))
    return records, [path.name for path in paths]


def download_records(temp_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    request = urllib.request.Request(UPSTREAM_API, headers={"User-Agent": "ContrAgent-investigation"})
    with urllib.request.urlopen(request, timeout=60) as response:
        files = json.load(response)
    records: list[dict[str, Any]] = []
    names: list[str] = []
    for item in files:
        name = item["name"]
        request = urllib.request.Request(RAW_BASE + item["path"], headers={"User-Agent": "ContrAgent-investigation"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        (temp_dir / name).write_bytes(data)
        records.extend(json.loads(data))
        names.append(name)
    return records, names


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    target_records = [
        record
        for record in records
        if (record.get("task") or {}).get("user_goal") == "transfer_funds"
        and not has_transfer_call(record)
    ]

    # Keep permitted and forbidden goals separate: no transfer is usually the
    # correct policy behavior for forbidden tasks, not a failure to complete.
    outcome_counts: Counter[str] = Counter()
    allowed_missed: list[dict[str, Any]] = []
    missing_eval = 0
    for record in target_records:
        task = record.get("task") or {}
        should_succeed = bool(task.get("action_should_succeed"))
        result = evaluator(record)
        eval_success = result.get("success")
        if eval_success is None:
            missing_eval += 1
            outcome_counts["Evaluation unavailable"] += 1
        elif should_succeed and eval_success:
            outcome_counts["Permitted task; evaluator success"] += 1
        elif should_succeed:
            outcome_counts["Permitted task; evaluator failure"] += 1
            allowed_missed.append(record)
        elif eval_success:
            outcome_counts["Forbidden task; evaluator success"] += 1
        else:
            outcome_counts["Forbidden task; evaluator failure"] += 1

    signals: Counter[str] = Counter()
    signal_denominators: Counter[str] = Counter()
    no_additional_signal = 0
    target_action_flag_false = 0
    for record in allowed_missed:
        result = evaluator(record)
        if result.get("action_called_correctly") is False:
            target_action_flag_false += 1
        has_other_signal = False
        for label, field in FAILURE_SIGNALS:
            value = result.get(field)
            if isinstance(value, bool):
                signal_denominators[label] += 1
                if not value:
                    signals[label] += 1
                    has_other_signal = True
        if not has_other_signal:
            no_additional_signal += 1

    return {
        "source": {
            "repository": "https://github.com/Leezekun/SOPBench",
            "commit": UPSTREAM_COMMIT,
            "domain": "bank",
        },
        "all_bank_records": len(records),
        "transfer_goal_records": sum(
            (r.get("task") or {}).get("user_goal") == "transfer_funds" for r in records
        ),
        "transfer_goal_records_without_transfer_call": len(target_records),
        "no_call_outcomes": dict(outcome_counts),
        "allowed_task_no_call_evaluator_failures": len(allowed_missed),
        "target_action_flag_false_among_allowed_task_failures": target_action_flag_false,
        "allowed_task_failures_with_no_other_false_evaluator_flag": no_additional_signal,
        "failure_signals_among_allowed_task_failures": {
            label: {
                "count": signals[label],
                "evaluable_records": signal_denominators[label],
            }
            for label, _ in FAILURE_SIGNALS
        },
        "records_without_evaluation": missing_eval,
        "interpretation": (
            "Failure signals are overlapping evaluator flags, not mutually exclusive root causes. "
            "No transfer on a forbidden task is not automatically a failure."
        ),
    }


def plot(summary: dict[str, Any], output_path: Path) -> None:
    outcomes = summary["no_call_outcomes"]
    outcome_order = [
        "Forbidden task; evaluator success",
        "Forbidden task; evaluator failure",
        "Permitted task; evaluator failure",
        "Permitted task; evaluator success",
        "Evaluation unavailable",
    ]
    colors = {
        "Forbidden task; evaluator success": "#4c956c",
        "Forbidden task; evaluator failure": "#b7c9bc",
        "Permitted task; evaluator failure": "#c8553d",
        "Permitted task; evaluator success": "#e6a15c",
        "Evaluation unavailable": "#929292",
    }
    signals = summary["failure_signals_among_allowed_task_failures"]
    signal_items = [(label, signals[label]["count"], signals[label]["evaluable_records"]) for label, _ in FAILURE_SIGNALS]
    signal_items.append(("No other failure flag", summary["allowed_task_failures_with_no_other_false_evaluator_flag"], summary["allowed_task_no_call_evaluator_failures"]))
    signal_items.sort(key=lambda row: row[1])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": "#263238",
        "axes.labelcolor": "#263238",
        "xtick.color": "#455a64",
        "ytick.color": "#263238",
        "axes.edgecolor": "#c8d0d3",
        "svg.fonttype": "none",
    })
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 6.6), gridspec_kw={"width_ratios": [1.08, 1.0]}
    )
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1)
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#c8d0d3")
        ax.tick_params(axis="y", length=0, pad=9)
        ax.grid(axis="x", color="#dce2e4", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    present_outcomes = [(name, outcomes.get(name, 0)) for name in outcome_order if outcomes.get(name, 0)]
    labels = [name for name, _ in present_outcomes]
    values = [value for _, value in present_outcomes]
    y = list(range(len(labels)))
    ax1.barh(y, values, color=[colors[name] for name in labels], height=0.62, zorder=3)
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Records")
    ax1.set_title("What no transfer means", loc="left", fontsize=13, fontweight="bold", pad=16)
    ax1.set_xlim(0, max(values, default=1) * 1.22)
    for yi, value in zip(y, values):
        ax1.text(value + max(values, default=1) * 0.025, yi, f"{value:,}", va="center", fontsize=10, fontweight="bold")

    s_labels = [item[0] for item in signal_items]
    s_values = [item[1] for item in signal_items]
    sy = list(range(len(s_labels)))
    signal_colors = ["#929292" if label == "No other failure flag" else "#c8553d" for label in s_labels]
    ax2.barh(sy, s_values, color=signal_colors, height=0.62, zorder=3)
    ax2.set_yticks(sy, s_labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Records")
    ax2.set_title("Other evaluator flags among 53 missed allowed tasks", loc="left", fontsize=11.5, fontweight="bold", pad=16)
    ax2.set_xlim(0, max(s_values, default=1) * 1.3 if any(s_values) else 1)
    for yi, (value, denom) in enumerate(zip(s_values, [item[2] for item in signal_items])):
        ax2.text(value + max(s_values, default=1) * 0.03, yi, f"{value:,} / {denom:,}", va="center", fontsize=9, fontweight="bold")

    total_no_call = summary["transfer_goal_records_without_transfer_call"]
    fig.suptitle(
        f"Bank SOPBench · {total_no_call:,} transfer-goal runs without a transfer call",
        x=0.06, y=0.98, ha="left", fontsize=16, fontweight="bold", color="#1f3037",
    )
    fig.text(
        0.06, 0.925,
        "A no-call on a forbidden task may be correct refusal; only missed permitted tasks are counted as completion failures.",
        ha="left", fontsize=9.5, color="#52656d",
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.13, wspace=0.62)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", transparent=False, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT, help="Directory containing bank output JSON files")
    parser.add_argument("--download", action="store_true", help="Download pinned upstream bank output files to a temporary directory")
    parser.add_argument("--svg", type=Path, default=HERE / "transfer-no-call-reasons.svg", help="SVG output path")
    parser.add_argument("--summary", type=Path, default=HERE / "transfer-no-call-summary.json", help="JSON summary output path")
    args = parser.parse_args()

    if args.download:
        with tempfile.TemporaryDirectory(prefix="sopbench-bank-") as temp:
            records, files = download_records(Path(temp))
            summary = analyze(records)
    else:
        records, files = read_records(args.input_dir)
        summary = analyze(records)

    summary["input_files"] = len(files)
    summary["input_file_names"] = files
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot(summary, args.svg)
    print(json.dumps({
        "bank_records": summary["all_bank_records"],
        "transfer_goal_no_calls": summary["transfer_goal_records_without_transfer_call"],
        "outcomes": summary["no_call_outcomes"],
        "missed_permitted_tasks": summary["allowed_task_no_call_evaluator_failures"],
        "svg": str(args.svg),
        "summary": str(args.summary),
    }, indent=2))


if __name__ == "__main__":
    main()
