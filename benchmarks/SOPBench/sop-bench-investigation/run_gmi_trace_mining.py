#!/usr/bin/env python3
"""Run a domain miner through the shared GMI client and save comparable results.

Example:
  python run_gmi_trace_mining.py --miner mine_healthcare_login_order \
      --dataset healthcare-login-order-dataset.json --model MODEL_ID
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gmi_trace_client import generate_completion, load_local_env, request_json, safe_name  # noqa: E402
from trace_mining_common import extract_formula, metrics  # noqa: E402


def _evaluate(miner: Any, dataset: dict[str, Any], formula: str) -> dict[str, Any]:
    if hasattr(miner, "evaluate"):
        return miner.evaluate(dataset, formula)
    train, error = miner.evaluate_candidate(formula, dataset["training"])
    if error:
        return {"parse_error": error, "candidate_formula": formula}
    heldout = dataset.get("heldout_active_gate", [])
    rows, error = miner.evaluate_candidate(formula, heldout)
    if error:
        return {"parse_error": error, "candidate_formula": formula}
    return {
        "candidate_formula": formula,
        "evaluated_sets": {
            "training": {"metrics": metrics(train), "rows": train},
            "heldout_active_gate": {"metrics": metrics(rows), "rows": rows},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miner", default="mine_healthcare_login_order", help="Importable domain miner module")
    parser.add_argument("--dataset", type=Path, default=HERE / "healthcare-login-order-dataset.json")
    parser.add_argument("--prompt-output", type=Path)
    parser.add_argument("--results-dir", type=Path, default=HERE)
    parser.add_argument("--prefix", default="healthcare-login-order")
    parser.add_argument("--model", action="append", help="GMI model ID; repeat to run several")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Completion budget, including model reasoning tokens")
    args = parser.parse_args()

    load_local_env(HERE.parents[2])
    key = os.environ.get("GMI_API_KEY")
    if not key:
        raise SystemExit("Set GMI_API_KEY in the environment or repository-root .env")
    catalog = request_json("/models", key, timeout=45).get("data", [])
    available = {item.get("id"): item for item in catalog if item.get("id")}
    if args.list_models:
        print("\n".join(sorted(available)))
        return
    selected = args.model or list(available)
    missing = [model for model in selected if model not in available]
    if missing:
        raise SystemExit("Unavailable model IDs: " + ", ".join(missing))

    miner = importlib.import_module(args.miner)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    prompt = miner.build_prompt(dataset)
    prompt_path = args.prompt_output or args.results_dir / f"{args.prefix}-miner-prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for model_id in selected:
        display = (available.get(model_id) or {}).get("name") or model_id
        print(f"Running {display} …", flush=True)
        started = time.monotonic()
        try:
            response = generate_completion(model_id, prompt, key, timeout=args.timeout, max_tokens=args.max_tokens)
            choice = (response.get("choices") or [{}])[0]
            raw = ((choice.get("message") or {}).get("content") or "").strip()
            formula = extract_formula(raw)
            evaluation = _evaluate(miner, dataset, formula)
            status = "unparseable" if evaluation.get("parse_error") else "completed"
            result = {
                "domain": dataset.get("domain", args.prefix),
                "provider": "GMI Cloud",
                "model_id": model_id,
                "display_name": display,
                "prompt_file": prompt_path.name,
                "dataset_file": args.dataset.name,
                "training_examples": len(dataset.get("training", [])),
                "heldout_sent_to_model": False,
                "request": {"temperature": 0, "max_tokens": args.max_tokens},
                "candidate_raw": raw,
                "candidate_formula": formula,
                "usage": response.get("usage", {}),
                "finish_reason": choice.get("finish_reason"),
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "evaluation": evaluation,
            }
            path = args.results_dir / f"{args.prefix}-result-{safe_name(model_id)}.json"
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            summary.append(result)
            held = evaluation.get("evaluated_sets", {}).get("heldout_active_gate", {}).get("metrics", {})
            csv_rows.append({"model_id": model_id, "display_name": display, "status": status,
                             "heldout_correct": held.get("correct", ""), "heldout_n": held.get("n", ""),
                             "heldout_accuracy": held.get("accuracy", ""), "candidate_formula": formula,
                             "error": evaluation.get("parse_error", "")})
            print(f"{display}: {held.get('correct', 'unparseable')}/{held.get('n', '?')} held-out", flush=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:240]}"
            path = args.results_dir / f"{args.prefix}-error-{safe_name(model_id)}.json"
            path.write_text(json.dumps({"model_id": model_id, "display_name": display, "status": "error", "error": error}, indent=2) + "\n", encoding="utf-8")
            csv_rows.append({"model_id": model_id, "display_name": display, "status": "error", "error": error})
            print(f"{display}: {error}", flush=True)

    # Rebuild aggregate artifacts from per-model files so incremental calls do
    # not erase earlier model runs.
    all_results = []
    all_rows = []
    for result_path in sorted(args.results_dir.glob(f"{args.prefix}-result-*.json")):
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        all_results.append(saved)
        held = saved.get("evaluation", {}).get("evaluated_sets", {}).get("heldout_active_gate", {}).get("metrics", {})
        all_rows.append({"model_id": saved.get("model_id", ""), "display_name": saved.get("display_name", ""),
                         "status": "unparseable" if saved.get("evaluation", {}).get("parse_error") else "completed",
                         "heldout_correct": held.get("correct", ""), "heldout_n": held.get("n", ""),
                         "heldout_accuracy": held.get("accuracy", ""), "candidate_formula": saved.get("candidate_formula", ""),
                         "error": saved.get("evaluation", {}).get("parse_error", "")})
    for error_path in sorted(args.results_dir.glob(f"{args.prefix}-error-*.json")):
        saved = json.loads(error_path.read_text(encoding="utf-8"))
        all_rows.append({"model_id": saved.get("model_id", ""), "display_name": saved.get("display_name", ""),
                         "status": "error", "error": saved.get("error", "")})
    (args.results_dir / f"{args.prefix}-results.json").write_text(
        json.dumps({"miner": args.miner, "dataset": args.dataset.name, "prompt": prompt_path.name, "results": all_results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.results_dir / f"{args.prefix}-results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["model_id", "display_name", "status", "heldout_correct", "heldout_n", "heldout_accuracy", "candidate_formula", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
