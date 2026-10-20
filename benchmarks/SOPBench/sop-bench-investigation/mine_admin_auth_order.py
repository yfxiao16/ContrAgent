#!/usr/bin/env python3
"""Run a small synthetic trace-mining pilot for admin-auth ordering.

Training labels describe only whether a successful transfer follows a
successful admin authentication while that task's gate is active. The hidden
reference formula is used only after candidate generation to grade the traces;
it is never included in the miner prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from contragent import Trace  # noqa: E402
from trace_mining_common import (  # noqa: E402
    evaluate_candidate as evaluate_formula,
    extract_formula,
    format_training_prompt,
    metrics,
    ordering_label,
    to_contragent_trace,
)


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = HERE / "admin-auth-order-mining-dataset.json"
PROMPT_PATH = HERE / "admin-auth-order-miner-prompt.txt"
RESULT_PATH = HERE / "admin-auth-order-mining-results.json"

# This reference is deliberately isolated from build_prompt(). It is used only
# by the post-generation evaluator and never serialized into the miner prompt.
REFERENCE_FORMULA = (
    "(!(called(transfer_funds) & arg_value(transfer_funds, succeeded) >= 1 & "
    "arg_value(transfer_funds, gate_authenticated_admin_password_active) >= 1) U "
    "(called(authenticate_admin_password) & "
    "arg_value(authenticate_admin_password, succeeded) >= 1)) | "
    "G(!(called(transfer_funds) & arg_value(transfer_funds, succeeded) >= 1 & "
    "arg_value(transfer_funds, gate_authenticated_admin_password_active) >= 1))"
)


def ev(tool: str, succeeded: bool = True) -> dict[str, Any]:
    return {"tool": tool, "succeeded": succeeded}


def trace(trace_id: str, label: str, events: list[dict[str, Any]], gate_active: bool = True) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "gate_active": gate_active,
        "label": label,
        "events": events,
    }


def make_dataset() -> dict[str, Any]:
    train_conforming = [
        [ev("authenticate_admin_password"), ev("transfer_funds"), ev("exit_conversation")],
        [ev("internal_check_username_exist"), ev("authenticate_admin_password"), ev("transfer_funds"), ev("exit_conversation")],
        [ev("authenticate_admin_password", False), ev("authenticate_admin_password"), ev("get_account_balance"), ev("transfer_funds"), ev("exit_conversation")],
        [ev("login_user"), ev("authenticate_admin_password"), ev("transfer_funds"), ev("send_receipt"), ev("exit_conversation")],
        [ev("get_account_balance"), ev("authenticate_admin_password"), ev("internal_check_username_exist"), ev("transfer_funds"), ev("exit_conversation")],
        [ev("authenticate_admin_password"), ev("login_user"), ev("transfer_funds"), ev("exit_conversation")],
        [ev("internal_check_username_exist"), ev("get_account_balance"), ev("authenticate_admin_password"), ev("transfer_funds"), ev("exit_conversation")],
        [ev("authenticate_admin_password", False), ev("login_user"), ev("authenticate_admin_password"), ev("transfer_funds"), ev("exit_conversation")],
        [ev("login_user"), ev("get_account_balance"), ev("authenticate_admin_password"), ev("send_receipt"), ev("transfer_funds"), ev("exit_conversation")],
        [ev("authenticate_admin_password"), ev("internal_check_username_exist"), ev("get_account_balance"), ev("transfer_funds"), ev("exit_conversation")],
    ]
    train_violating = [
        [ev("transfer_funds"), ev("authenticate_admin_password"), ev("exit_conversation")],
        [ev("get_account_balance"), ev("transfer_funds"), ev("authenticate_admin_password"), ev("exit_conversation")],
        [ev("authenticate_admin_password", False), ev("transfer_funds"), ev("authenticate_admin_password"), ev("exit_conversation")],
        [ev("login_user"), ev("transfer_funds"), ev("get_account_balance"), ev("authenticate_admin_password"), ev("exit_conversation")],
        [ev("internal_check_username_exist"), ev("authenticate_admin_password", False), ev("transfer_funds"), ev("authenticate_admin_password"), ev("exit_conversation")],
        [ev("get_account_balance"), ev("login_user"), ev("transfer_funds"), ev("send_receipt"), ev("authenticate_admin_password"), ev("exit_conversation")],
        [ev("authenticate_admin_password", False), ev("login_user"), ev("get_account_balance"), ev("transfer_funds"), ev("authenticate_admin_password"), ev("exit_conversation")],
        [ev("send_receipt"), ev("transfer_funds"), ev("internal_check_username_exist"), ev("authenticate_admin_password"), ev("exit_conversation")],
        [ev("login_user"), ev("authenticate_admin_password", False), ev("transfer_funds"), ev("get_account_balance"), ev("authenticate_admin_password"), ev("exit_conversation")],
        [ev("get_account_balance"), ev("internal_check_username_exist"), ev("transfer_funds"), ev("authenticate_admin_password"), ev("send_receipt"), ev("exit_conversation")],
    ]
    heldout_conforming = [
        [ev("authenticate_admin_password"), ev("authenticate_admin_password", False), ev("transfer_funds"), ev("exit_conversation")],
        [ev("get_account_balance"), ev("authenticate_admin_password", False), ev("authenticate_admin_password"), ev("send_receipt"), ev("transfer_funds"), ev("exit_conversation")],
        [ev("login_user"), ev("internal_check_username_exist"), ev("authenticate_admin_password"), ev("get_account_balance"), ev("transfer_funds"), ev("exit_conversation")],
        [ev("authenticate_admin_password"), ev("send_receipt"), ev("login_user"), ev("transfer_funds"), ev("exit_conversation")],
        [ev("get_account_balance"), ev("authenticate_admin_password"), ev("send_receipt"), ev("internal_check_username_exist"), ev("transfer_funds"), ev("exit_conversation")],
    ]
    heldout_violating = [
        [ev("transfer_funds"), ev("send_receipt"), ev("authenticate_admin_password"), ev("exit_conversation")],
        [ev("authenticate_admin_password", False), ev("get_account_balance"), ev("transfer_funds"), ev("login_user"), ev("authenticate_admin_password"), ev("exit_conversation")],
        [ev("login_user"), ev("transfer_funds"), ev("authenticate_admin_password", False), ev("authenticate_admin_password"), ev("exit_conversation")],
        [ev("internal_check_username_exist"), ev("get_account_balance"), ev("transfer_funds"), ev("authenticate_admin_password"), ev("exit_conversation")],
        [ev("send_receipt"), ev("transfer_funds"), ev("authenticate_admin_password", False), ev("authenticate_admin_password"), ev("exit_conversation")],
    ]
    inactive_controls = [
        trace("inactive_no_auth_1", "conforms", [ev("transfer_funds"), ev("exit_conversation")], False),
        trace("inactive_no_auth_2", "conforms", [ev("get_account_balance"), ev("transfer_funds"), ev("exit_conversation")], False),
        trace("inactive_auth_after_1", "conforms", [ev("transfer_funds"), ev("authenticate_admin_password"), ev("exit_conversation")], False),
        trace("inactive_auth_after_2", "conforms", [ev("send_receipt"), ev("transfer_funds"), ev("authenticate_admin_password"), ev("exit_conversation")], False),
    ]
    active_no_transfer = [
        trace("active_no_transfer_1", "conforms", [ev("get_account_balance"), ev("exit_conversation")], True),
        trace("active_no_transfer_2", "conforms", [ev("authenticate_admin_password"), ev("exit_conversation")], True),
    ]
    train = [
        trace(f"train_conforms_{i + 1:02}", "conforms", events)
        for i, events in enumerate(train_conforming)
    ] + [
        trace(f"train_violates_{i + 1:02}", "violates", events)
        for i, events in enumerate(train_violating)
    ]
    heldout = [
        trace(f"heldout_conforms_{i + 1:02}", "conforms", events)
        for i, events in enumerate(heldout_conforming)
    ] + [
        trace(f"heldout_violates_{i + 1:02}", "violates", events)
        for i, events in enumerate(heldout_violating)
    ]
    return {
        "dataset_type": "synthetic controlled trace-mining pilot",
        "target_rule_description": "If the admin-auth gate is active, a successful transfer must be preceded by successful admin-password authentication.",
        "label_definition": "Labels indicate conformance to the target rule, not whether the task itself is permitted or the bank action should succeed.",
        "trace_schema": {"tool": "tool-call name", "succeeded": "Boolean tool result", "gate_active": "task-level SOP context; active on every miner example"},
        "training": train,
        "heldout_active_gate": heldout,
        "heldout_inactive_gate_controls": inactive_controls,
        "heldout_active_gate_no_transfer_controls": active_no_transfer,
        "counts": {
            "training": {"conforms": len(train_conforming), "violates": len(train_violating)},
            "heldout_active_gate": {"conforms": len(heldout_conforming), "violates": len(heldout_violating)},
            "heldout_inactive_gate_controls": len(inactive_controls),
            "heldout_active_gate_no_transfer_controls": len(active_no_transfer),
        },
    }


def build_prompt(dataset: dict[str, Any]) -> str:
    # Only training traces and their labels enter the model prompt. No reference
    # formula, held-out trace, or evaluation result is exposed here.
    examples = [
        {
            "gate_active": item["gate_active"],
            "label": item["label"],
            "events": [event for event in item["events"] if event["tool"] != "exit_conversation"],
            "trace_ends_after_last_event": True,
        }
        for item in dataset["training"]
    ]
    instructions = (
        "Infer one concise LTLf contract from these labeled traces. Each trace is from a task "
        "whose authenticated_admin_password gate is active. Labels mean whether the trace "
        "conforms to the hidden rule. Tool events are ordered; each has a tool name and a "
        "Boolean succeeded result. Each JSON trace omits the explicit exit_conversation "
        "marker and ends after its final listed event. Ignore unrelated calls unless they affect the rule. "
        "Write a contract that generalizes beyond these exact sequences. The contract language "
        "supports called(tool), arg_value(tool, field), comparisons such as >= 1, Boolean "
        "operators !, &, |, ->, and temporal operators G, F, U. The transfer call carries "
        "gate_authenticated_admin_password_active=1 when its gate is active. "
        "Syntax example only (an unrelated rule): "
        "G((called(send_receipt) & arg_value(send_receipt, succeeded) >= 1) -> called(get_order_details)). "
        "Use called(tool) and arg_value(tool, field) predicates; do not use bare tool names, "
        "assignments with =, or the word 'succeed' as a predicate. Ignore unrelated calls "
        "that vary independently of the labels. "
        "Return exactly one line in this form: G: <formula>. Do not include explanation or code fences."
    )
    return format_training_prompt(instructions, examples)


def load_candidate(model_id: str, prompt: str, max_new_tokens: int, dtype_name: str) -> tuple[str, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.eval()
    messages = [
        {"role": "system", "content": "You infer formal temporal contracts from labeled interaction traces."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del model
    return generated, extract_formula(generated)


def to_trace(item: dict[str, Any]) -> Trace:
    return to_contragent_trace(
        item,
        agent_id="bank_assistant",
        goal_tool="transfer_funds",
        gate_arg="gate_authenticated_admin_password_active",
    )


def reference_label(item: dict[str, Any]) -> str:
    return ordering_label(
        item,
        prerequisite_tool="authenticate_admin_password",
        goal_tool="transfer_funds",
    )


def evaluate_candidate(formula_text: str, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    # Full traces are evaluated with finite-trace liveness resolved at end.
    return evaluate_formula(
        formula_text,
        items,
        agent_id="bank_assistant",
        goal_tool="transfer_funds",
        prerequisite_tool="authenticate_admin_password",
        gate_arg="gate_authenticated_admin_password_active",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face causal language model ID")
    parser.add_argument("--candidate", type=Path, help="Use a saved candidate line instead of model inference")
    parser.add_argument("--results", type=Path, default=RESULT_PATH, help="Where to write evaluation results")
    parser.add_argument("--max-new-tokens", type=int, default=280)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--refresh-data", action="store_true", help="Regenerate the fixed synthetic trace dataset")
    args = parser.parse_args()

    if args.refresh_data or not DATA_PATH.exists():
        dataset = make_dataset()
        DATA_PATH.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    else:
        dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    prompt = build_prompt(dataset)
    PROMPT_PATH.write_text(prompt, encoding="utf-8")
    if args.candidate:
        generated = args.candidate.read_text(encoding="utf-8").strip()
        formula_text = extract_formula(generated)
        model_id = None
    else:
        generated, formula_text = load_candidate(args.model, prompt, args.max_new_tokens, args.dtype)
        model_id = args.model

    training = dataset["training"]
    heldout = dataset["heldout_active_gate"]
    inactive = dataset["heldout_inactive_gate_controls"]
    no_transfer = dataset["heldout_active_gate_no_transfer_controls"]
    all_eval = heldout + inactive + no_transfer
    evaluated, parse_error = evaluate_candidate(formula_text, all_eval)
    if parse_error:
        result = {
            "model_id": model_id,
            "dtype": None if args.candidate else args.dtype,
            "candidate_raw": generated,
            "candidate_formula": formula_text,
            "parse_error": parse_error,
            "pair_check": "UNPARSEABLE",
            "prompt_path": str(PROMPT_PATH.name),
            "dataset_path": str(DATA_PATH.name),
            "reference_formula_used_only_for_evaluation": REFERENCE_FORMULA,
        }
    else:
        by_id = {r["trace_id"]: r for r in evaluated}
        train_eval, train_error = evaluate_candidate(formula_text, training)
        pair_ok = (
            by_id["heldout_conforms_01"]["predicted"] == "conforms"
            and by_id["heldout_violates_01"]["predicted"] == "violates"
        )
        inactive_ok = all(by_id[x["trace_id"]]["predicted"] == "conforms" for x in inactive)
        result = {
            "model_id": model_id,
            "dtype": None if args.candidate else args.dtype,
            "candidate_raw": generated,
            "candidate_formula": formula_text,
            "parse_error": train_error,
            "training_metrics": metrics(train_eval),
            "heldout_active_gate_metrics": metrics([by_id[x["trace_id"]] for x in heldout]),
            "inactive_gate_control_pass_rate": sum(by_id[x["trace_id"]]["predicted"] == "conforms" for x in inactive) / len(inactive),
            "active_gate_no_transfer_control_pass_rate": sum(by_id[x["trace_id"]]["predicted"] == "conforms" for x in no_transfer) / len(no_transfer),
            "pair_check": "PASS" if pair_ok and inactive_ok else "FAIL",
            "pair_and_control_checks": {"auth_before_transfer_accepted": pair_ok, "inactive_gate_controls_accepted": inactive_ok},
            "evaluated_traces": evaluated,
            "prompt_path": PROMPT_PATH.name,
            "dataset_path": DATA_PATH.name,
            "reference_formula_used_only_for_evaluation": REFERENCE_FORMULA,
        }
    args.results.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result.get("pair_check") != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
