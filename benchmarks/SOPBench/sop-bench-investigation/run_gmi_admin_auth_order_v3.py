#!/usr/bin/env python3
"""Run the scoped v3 prompt on GMI models and score complete traces.

Set GMI_API_KEY in the environment or place it in the repository-root .env.
The key is never written to result files or printed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import mine_admin_auth_order as miner  # noqa: E402
from gmi_trace_client import generate_completion, load_local_env, request_json, safe_name  # noqa: E402

PROMPT_PATH = HERE / "admin-auth-order-miner-prompt-v3-scoped-gate.txt"
DATA_PATH = HERE / "admin-auth-order-mining-dataset-v2-20-per-class.json"
CANDIDATES = {
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "deepseek-ai/DeepSeek-V4-Flash": "DeepSeek V4 Flash",
    "Qwen/Qwen3.8-Max": "Qwen3.8 Max",
    "openai/gpt-oss-120b": "GPT OSS 120B",
    "Qwen/Qwen3.5-397B-A17B": "Qwen3.5 397B A17B",
    "openai/gpt-6-astra": "GPT-6 Astra",
    "openai/gpt-5.4-nano": "GPT-5.4 Nano",
    "Qwen/Qwen3.5-27B": "Qwen3.5 27B",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": "Nemotron 3.5 Lightning 30B A3B",
    "stepfun-ai/Step-3.7-Flash": "Step-3.7 Flash",
    "zai-org/GLM-5.3-Flash": "GLM-5.3 Flash",
    "deepseek-ai/DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "openai/gpt-5.6-terra": "GPT-5.6 Terra",
    "moonshotai/Kimi-K2.6": "Kimi K2.6",
    "x-ai/grok-4.5": "Grok 4.5",
}

def summarize(model_id: str, display_name: str, prompt: str, generated: str, usage: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
    formula_text = miner.extract_formula(generated)
    raw_result = {
        "provider": "GMI Cloud",
        "model_id": model_id,
        "display_name": display_name,
        "prompt_file": PROMPT_PATH.name,
        "dataset_file": DATA_PATH.name,
        "training_examples": 40,
        "heldout_sent_to_model": False,
        "request": {"temperature": 0, "max_tokens": 2048},
        "candidate_raw": generated,
        "usage": usage,
        "finish_reason": finish_reason,
    }
    result_path = HERE / f"admin-auth-order-mining-result-v3-{safe_name(model_id)}.json"
    result_path.write_text(json.dumps(raw_result, indent=2) + "\n", encoding="utf-8")
    stale_error = HERE / f"admin-auth-order-run-v3-{safe_name(model_id)}.json"
    if stale_error.exists():
        stale_error.unlink()

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    all_controls = data["heldout_controls"]
    try:
        miner.parse_repr(formula_text)
    except Exception as exc:
        evaluation = {
            "model_id": model_id,
            "display_name": display_name,
            "candidate_formula": formula_text,
            "parse_error": f"{type(exc).__name__}: {exc}",
            "evaluation_semantics": "complete finite traces (include_liveness=True)",
            "pair_check": "UNPARSEABLE",
            "prompt_path": PROMPT_PATH.name,
            "dataset_path": DATA_PATH.name,
        }
    else:
        train, _ = miner.evaluate_candidate(formula_text, data["training"])
        held, _ = miner.evaluate_candidate(formula_text, data["heldout_active_gate"])
        controls, _ = miner.evaluate_candidate(formula_text, all_controls)
        inactive_items = [x for x in all_controls if not x["gate_active"]]
        active_items = [x for x in all_controls if x["gate_active"]]
        by_id = {x["trace_id"]: x for x in controls}
        inactive_rows = [by_id[x["trace_id"]] for x in inactive_items]
        active_rows = [by_id[x["trace_id"]] for x in active_items]
        pair_ok = (
            next(x for x in held if x["trace_id"] == "holdout_conforms_01")["predicted"] == "conforms"
            and next(x for x in held if x["trace_id"] == "holdout_violates_01")["predicted"] == "violates"
        )
        evaluation = {
            "model_id": model_id,
            "display_name": display_name,
            "candidate_formula": formula_text,
            "evaluation_semantics": "complete finite traces (include_liveness=True)",
            "training_metrics": miner.metrics(train),
            "heldout_active_gate_metrics": miner.metrics(held),
            "inactive_gate_control_pass_rate": sum(x["predicted"] == "conforms" for x in inactive_rows) / len(inactive_rows),
            "active_gate_control_pass_rate": sum(x["predicted"] == "conforms" for x in active_rows) / len(active_rows),
            "pair_check": "PASS" if pair_ok else "FAIL",
            "pair_and_control_checks": {"auth_before_transfer_accepted": pair_ok,
                                        "inactive_gate_controls_accepted": all(x["predicted"] == "conforms" for x in inactive_rows)},
            "prompt_path": PROMPT_PATH.name,
            "dataset_path": DATA_PATH.name,
        }
    eval_path = HERE / f"admin-auth-order-mining-evaluation-v3-{safe_name(model_id)}.json"
    eval_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    return {"model": display_name, "candidate": formula_text, "eval": evaluation, "raw_file": result_path.name, "evaluation_file": eval_path.name}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true", help="Print available model IDs without making inference calls")
    parser.add_argument("--model", action="append", help="Model ID to run; repeat this option. Defaults to prior untested hosted candidates.")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    load_local_env(HERE.parents[2])
    key = os.environ.get("GMI_API_KEY")
    if not key:
        raise SystemExit("Set GMI_API_KEY in the environment; the key is not stored in this script.")
    catalog = request_json("/models", key, timeout=45).get("data", [])
    available = {m.get("id"): m for m in catalog if m.get("id")}
    if args.list_models:
        print("\n".join(sorted(available)))
        return
    selected = args.model or [m for m in CANDIDATES if m in available]
    missing = [m for m in selected if m not in available]
    if missing:
        print("Unavailable model IDs: " + ", ".join(missing))
    if not selected:
        raise SystemExit("No selected prior candidates are available in the current account model list.")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    successes = []
    attempts = []
    for model_id in selected:
        display = CANDIDATES.get(model_id, model_id)
        print(f"Running {display} …", flush=True)
        started = time.monotonic()
        try:
            response = generate_completion(model_id, prompt, key, timeout=args.timeout)
            choice = (response.get("choices") or [{}])[0]
            generated = ((choice.get("message") or {}).get("content") or "").strip()
            formula_text = miner.extract_formula(generated)
            usage = response.get("usage", {})
            result = summarize(model_id, display, prompt, generated, usage, choice.get("finish_reason"))
            result["elapsed_seconds"] = round(time.monotonic() - started, 1)
            successes.append(result)
            evaluation = result["eval"]
            attempts.append({"model_id": model_id, "display_name": display, "status": "completed", "evaluation": evaluation, "candidate_formula": formula_text})
            metrics = evaluation.get("heldout_active_gate_metrics", {})
            print(f"Completed {display}: {metrics.get('correct', 'unparseable')}/{metrics.get('n', '?')} active-gate holdout; inactive controls pass={evaluation.get('inactive_gate_control_pass_rate', 'n/a')}", flush=True)
        except Exception as exc:
            failure = {"model_id": model_id, "display_name": display, "status": "error", "error": f"{type(exc).__name__}: {str(exc)[:240]}", "elapsed_seconds": round(time.monotonic() - started, 1)}
            failure_path = HERE / f"admin-auth-order-run-v3-{safe_name(model_id)}.json"
            failure_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
            attempts.append({"model_id": model_id, "display_name": display, "status": "error", "error": failure["error"]})
            print(f"Failed {display}: {failure['error']}", flush=True)
    summary_path = HERE / "admin-auth-order-model-comparison-v3-followup.json"
    summary_path.write_text(json.dumps({"prompt": PROMPT_PATH.name, "training_examples": 40, "heldout_sent_to_model": False, "results": successes}, indent=2) + "\n", encoding="utf-8")
    csv_path = HERE / "admin-auth-order-model-comparison-v3-followup.csv"
    csv_rows: dict[str, dict[str, Any]] = {}
    for eval_file in HERE.glob("admin-auth-order-mining-evaluation-v3-*.json"):
        saved = json.loads(eval_file.read_text(encoding="utf-8"))
        held = saved.get("heldout_active_gate_metrics", {})
        csv_rows[saved["model_id"]] = {"model_id": saved["model_id"], "display_name": saved.get("display_name", saved["model_id"]), "status": "unparseable" if "parse_error" in saved else "completed", "holdout_correct": held.get("correct", ""), "holdout_n": held.get("n", ""), "holdout_accuracy": held.get("accuracy", ""), "inactive_gate_pass_rate": saved.get("inactive_gate_control_pass_rate", ""), "active_gate_pass_rate": saved.get("active_gate_control_pass_rate", ""), "pair_check": saved.get("pair_check", ""), "candidate_formula": saved.get("candidate_formula", ""), "error": saved.get("parse_error", "")}
    for error_file in HERE.glob("admin-auth-order-run-v3-*.json"):
        saved = json.loads(error_file.read_text(encoding="utf-8"))
        csv_rows[saved["model_id"]] = {"model_id": saved["model_id"], "display_name": saved.get("display_name", saved["model_id"]), "status": "error", "error": saved.get("error", "")}
    for item in attempts:
        model_id = item["model_id"]
        if model_id in csv_rows:
            evaluation = item.get("evaluation", {})
            csv_rows[model_id]["status"] = "unparseable" if "parse_error" in evaluation else item["status"]
            if "error" in item:
                csv_rows[model_id]["error"] = item["error"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["model_id", "display_name", "status", "holdout_correct", "holdout_n", "holdout_accuracy", "inactive_gate_pass_rate", "active_gate_pass_rate", "pair_check", "candidate_formula", "error"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows[mid] for mid in sorted(csv_rows))
    print(f"Wrote {summary_path.name}")
    print(f"Wrote {csv_path.name}")

if __name__ == "__main__":
    main()
