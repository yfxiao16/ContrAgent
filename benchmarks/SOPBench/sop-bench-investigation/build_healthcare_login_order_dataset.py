#!/usr/bin/env python3
"""Build a sanitized healthcare ordering dataset from released SOPBench traces.

Only tool names, tool success flags, the task-level gate bit, and an opaque
task-group hash are written. Original prompts, arguments, task descriptions,
database values, and benchmark outcome labels are deliberately discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from trace_mining_common import ordering_label, result_succeeded


REPO_API = "https://api.github.com/repos/Leezekun/SOPBench/contents/output/healthcare"
AGENT_ID = "healthcare_assistant"
PREREQUISITE_TOOL = "login_user"
GOAL_TOOL = "update_policy"
RESET_TOOL = "logout_user"
GATE_NAME = "logged_in_user"
GATE_ARG = "gate_logged_in_user_active"


def _request_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "ContrAgent trace-mining study"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def _required_gates(node: Any) -> set[str]:
    """Match SOPBench-to-ContrAgent conservative gate extraction semantics."""
    if not isinstance(node, list) or not node:
        return set()
    operator = node[0]
    if operator == "single":
        name = node[1]
        if isinstance(name, str) and name.startswith("not "):
            name = name[len("not "):]
        return {name} if isinstance(name, str) else set()
    if operator in ("and", "chain"):
        result: set[str] = set()
        for child in node[1]:
            result |= _required_gates(child)
        return result
    if operator == "or":
        branches = [_required_gates(child) for child in node[1]]
        if not branches:
            return set()
        result = set(branches[0])
        for branch in branches[1:]:
            result &= branch
        return result
    if operator == "gate":
        return set()
    if operator == "not":
        return _required_gates(node[1])
    return set()


def _tool_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    interactions = record.get("interactions") or []
    messages = interactions[0].get("interaction", []) if interactions else []
    result_by_id = {
        message.get("tool_call_id"): message.get("content")
        for message in messages
        if message.get("tool_call_id")
    }
    events = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            tool = call.get("function", {}).get("name")
            if not tool or tool == "exit_conversation":
                continue
            events.append(
                {
                    "tool": tool,
                    "succeeded": result_succeeded(result_by_id.get(call.get("id"))),
                }
            )
    return events


def _group_id(task: dict[str, Any]) -> str:
    canonical = json.dumps(task, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _iter_records(source_dir: Path | None) -> tuple[list[tuple[str, list[dict[str, Any]]]], int]:
    if source_dir:
        paths = sorted(source_dir.glob("*.json"))
        records = []
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append((path.name, data if isinstance(data, list) else []))
        return records, len(paths)

    files = _request_json(REPO_API)
    files = [item for item in files if item.get("name", "").endswith(".json")]
    output = []
    for item in files:
        data = _request_json(item["download_url"])
        output.append((item["name"], data if isinstance(data, list) else []))
    return output, len(files)


def build_dataset(source_dir: Path | None = None) -> dict[str, Any]:
    file_records, files_seen = _iter_records(source_dir)
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    control_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()

    for filename, records in file_records:
        for record in records:
            counts["records_seen"] += 1
            setup = record.get("setup") or {}
            if setup.get("tool_list") != "full":
                counts["excluded_non_full_tool_setting"] += 1
                continue
            task = record.get("task") or {}
            if task.get("user_goal") != GOAL_TOOL:
                continue
            gate_active = GATE_NAME in _required_gates(task.get("constraints"))
            events = _tool_events(record)
            if any(event["tool"] == RESET_TOOL and event["succeeded"] for event in events):
                counts["excluded_successful_logout"] += 1
                continue
            group_id = _group_id(task)
            item = {
                "trace_id": "",
                "group_id": group_id,
                "gate_active": gate_active,
                "events": events,
                "source_setting": filename,
            }
            label = ordering_label(
                item,
                prerequisite_tool=PREREQUISITE_TOOL,
                goal_tool=GOAL_TOOL,
                reset_tool=RESET_TOOL,
            )
            item["label"] = label
            counts["eligible_update_policy_runs"] += 1
            counts[f"eligible_{label}"] += 1
            if gate_active:
                groups[group_id][label].append(item)
                if not any(e["tool"] == GOAL_TOOL and e["succeeded"] for e in events):
                    control_groups[group_id].append(item)
            else:
                counts["eligible_but_gate_inactive"] += 1

    # Allocate complete task configurations to one side before selecting
    # traces. This prevents another model's run of the same task leaking into
    # the held-out set.
    active_ids = sorted(groups, key=lambda group: _stable_key("healthcare-split:" + group))
    selected_eval_ids: list[str] = []
    for percentage in range(20, 41):
        count = max(1, round(len(active_ids) * percentage / 100))
        proposed = active_ids[:count]
        remaining = active_ids[count:]
        eval_class_counts = {
            label: sum(bool(groups[group].get(label)) for group in proposed)
            for label in ("conforms", "violates")
        }
        train_class_counts = {
            label: sum(bool(groups[group].get(label)) for group in remaining)
            for label in ("conforms", "violates")
        }
        if min(eval_class_counts.values()) >= 10 and min(train_class_counts.values()) >= 20:
            selected_eval_ids = proposed
            break
    if not selected_eval_ids:
        raise RuntimeError("Could not create a task-disjoint 20/10-per-class split from the source traces")
    eval_id_set = set(selected_eval_ids)

    def choose(group_ids: list[str], label: str, limit: int, split_name: str) -> list[dict[str, Any]]:
        candidates = []
        for group_id in group_ids:
            for item in groups[group_id].get(label, []):
                candidates.append((group_id, item))
        candidates.sort(key=lambda pair: _stable_key(f"{split_name}:{label}:{pair[0]}:{pair[1]['source_setting']}:{json.dumps(pair[1]['events'], sort_keys=True)}"))
        selected_by_group: dict[str, dict[str, Any]] = {}
        for group_id, raw in candidates:
            if group_id in selected_by_group:
                continue
            item = {k: v for k, v in raw.items() if k not in ("source_setting", "group_id")}
            item["trace_id"] = f"{split_name}_{label}_{len(selected_by_group) + 1:02d}"
            item["source_group"] = group_id
            item["source_setting"] = raw["source_setting"]
            selected_by_group[group_id] = item
            if len(selected_by_group) == limit:
                break
        if len(selected_by_group) != limit:
            raise RuntimeError(f"Need {limit} {label} traces for {split_name}; found {len(selected_by_group)} task groups")
        return list(selected_by_group.values())

    train_ids = [group for group in active_ids if group not in eval_id_set]
    training = choose(train_ids, "conforms", 20, "train") + choose(train_ids, "violates", 20, "train")
    heldout = choose(selected_eval_ids, "conforms", 10, "heldout") + choose(selected_eval_ids, "violates", 10, "heldout")

    controls = []
    control_candidates = []
    for group_id in selected_eval_ids:
        for item in control_groups.get(group_id, []):
            control_candidates.append((group_id, item))
    control_candidates.sort(key=lambda pair: _stable_key(f"healthcare-control:{pair[0]}:{pair[1]['source_setting']}"))
    seen_control_groups = set()
    for group_id, raw in control_candidates:
        if group_id in seen_control_groups:
            continue
        seen_control_groups.add(group_id)
        item = {k: v for k, v in raw.items() if k not in ("source_setting", "group_id")}
        item["trace_id"] = f"heldout_no_success_{len(controls) + 1:02d}"
        item["source_group"] = group_id
        item["source_setting"] = raw["source_setting"]
        controls.append(item)
        if len(controls) == 8:
            break

    return {
        "dataset_type": "sanitized recorded healthcare SOPBench traces",
        "source": "Leezekun/SOPBench public output/healthcare, full-tool settings",
        "source_files_seen": files_seen,
        "source_records_seen": counts["records_seen"],
        "filtered_run_counts": dict(counts),
        "domain": "healthcare",
        "agent_id": AGENT_ID,
        "rule": {
            "prerequisite_tool": PREREQUISITE_TOOL,
            "goal_tool": GOAL_TOOL,
            "reset_tool": RESET_TOOL,
            "gate_name": GATE_NAME,
            "gate_arg": GATE_ARG,
            "gate_active_on_training_and_holdout": True,
            "target_description": "When the task's logged_in_user gate is active, every successful update_policy call must be preceded by a successful login_user call that has not been cleared by a successful logout_user call.",
            "label_definition": "Trace conformance to this one ordering prerequisite only; labels do not mean the whole task was safe or correct.",
        },
        "split_method": "Opaque task-configuration hashes are assigned wholly to train or holdout before model runs are selected.",
        "excluded_from_mining_examples": [
            "successful-logout traces, to keep the first temporal-order pilot focused on login-before-goal without a session-reset case",
            "task and agent prompts, tool arguments, database values, evaluator fields, and safe/unsafe filename labels",
        ],
        "training": training,
        "heldout_active_gate": heldout,
        "heldout_active_gate_no_success_controls": controls,
        "heldout_inactive_gate_controls": [],
        "counts": {
            "training": {"conforms": 20, "violates": 20},
            "heldout_active_gate": {"conforms": 10, "violates": 10},
            "heldout_active_gate_no_success_controls": len(controls),
            "heldout_inactive_gate_controls": 0,
            "distinct_task_groups_train": len({x["source_group"] for x in training}),
            "distinct_task_groups_holdout": len({x["source_group"] for x in heldout}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Read downloaded healthcare JSON files from this directory instead of GitHub")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("healthcare-login-order-dataset.json"))
    args = parser.parse_args()
    dataset = build_dataset(args.source_dir)
    args.output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(dataset["counts"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
