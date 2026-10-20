#!/usr/bin/env python3
"""Build the healthcare prompt and score candidate LTLf rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trace_mining_common import evaluate_candidate as evaluate_formula
from trace_mining_common import extract_formula, format_training_prompt, metrics


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "healthcare-login-order-dataset.json"
PROMPT_PATH = HERE / "healthcare-login-order-miner-prompt.txt"


def build_prompt(dataset: dict[str, Any]) -> str:
    rule = dataset["rule"]
    examples = [
        {
            "gate_active": item["gate_active"],
            "label": item["label"],
            "events": item["events"],
            "trace_ends_after_last_event": True,
        }
        for item in dataset["training"]
    ]
    instruction = (
        "Infer one concise LTLf contract from labeled interaction traces. The traces are "
        "from healthcare tasks, and the task-level gate_active flag is trusted SOP context, "
        "not an event. Every training example has gate_active=true. The labels indicate "
        "conformance to one hidden procedural rule, not whether the whole task was safe, "
        "permitted, or completed correctly. Each event is an ordered tool call with a Boolean "
        "succeeded result. These examples omit the exit_conversation marker and omit runs "
        "with a successful logout_user call. Infer a concise rule that generalizes beyond "
        "the exact sequences; do not add obligations about unrelated tools.\n\n"
        "Allowed atoms: called(tool), arg_value(tool, field) >= 1.\n"
        "Allowed operators: !, &, |, ->, G, F, U. Use only these atoms and operators. "
        "For success, use arg_value(tool, succeeded) >= 1. The task gate is represented "
        f"on {rule['goal_tool']} events by arg_value({rule['goal_tool']}, {rule['gate_arg']}) >= 1.\n\n"
        "Syntax examples only; these describe unrelated rules:\n"
        "G: G((called(send_receipt) & arg_value(send_receipt, succeeded) >= 1) -> F(called(get_order_details)))\n"
        "G: (!(called(delete_archive)) U (called(approve_statement) & arg_value(approve_statement, succeeded) >= 1))\n\n"
        "Return exactly one line in this form: G: <formula>. Do not include explanation or code fences."
    )
    return format_training_prompt(instruction, examples)


def evaluate(dataset: dict[str, Any], formula_text: str) -> dict[str, Any]:
    rule = dataset["rule"]
    kwargs = {
        "agent_id": dataset["agent_id"],
        "goal_tool": rule["goal_tool"],
        "prerequisite_tool": rule["prerequisite_tool"],
        "gate_arg": rule["gate_arg"],
        "reset_tool": rule.get("reset_tool"),
    }
    groups = {"training": dataset["training"], **dataset.get("evaluation_sets", {})}
    if "evaluation_sets" not in dataset:
        groups["heldout_active_gate"] = dataset.get("heldout_active_gate", [])
        groups["heldout_controls"] = dataset.get("heldout_active_gate_no_success_controls", [])
        groups["heldout_controls"].extend(dataset.get("heldout_inactive_gate_controls", []))
    evaluated = {}
    for name, items in groups.items():
        rows, error = evaluate_formula(formula_text, items, **kwargs)
        if error:
            return {"parse_error": error, "candidate_formula": formula_text}
        evaluated[name] = {"metrics": metrics(rows), "rows": rows}
    return {"candidate_formula": formula_text, "evaluated_sets": evaluated}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="File containing a candidate formula")
    parser.add_argument("--dataset", type=Path, default=DATA_PATH)
    parser.add_argument("--prompt-output", type=Path, default=PROMPT_PATH)
    parser.add_argument("--results", type=Path, help="Optional result JSON output path")
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    prompt = build_prompt(dataset)
    args.prompt_output.write_text(prompt, encoding="utf-8")
    candidate = extract_formula(args.candidate.read_text(encoding="utf-8"))
    result = evaluate(dataset, candidate)
    result["prompt_file"] = args.prompt_output.name
    result["dataset_file"] = args.dataset.name
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.results:
        args.results.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
