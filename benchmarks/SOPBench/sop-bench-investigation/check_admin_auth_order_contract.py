#!/usr/bin/env python3
"""Evaluate a matched auth/transfer order pair with ContrAgent's bank contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from contragent import ContrAgent, Event, Trace, contract, load_config, parse_repr  # noqa: E402


BANK_CONFIG = REPO_ROOT / "contragent" / "contracts" / "sopbench" / "bank.yaml"
FIXTURE_PATH = HERE / "controlled-admin-auth-order-fixtures.json"
OUTPUT_PATH = HERE / "controlled-admin-auth-order-results.json"
CONTRACT_DESC = "transfer_funds: authenticated_admin_password"
GATE_ARG = "gate_authenticated_admin_password_active"


def bank_contract_formula() -> str:
    config = load_config(BANK_CONFIG)
    entries = [
        entry
        for agent in config.agents.values()
        for entry in agent.contracts
        if entry.desc == CONTRACT_DESC
    ]
    if len(entries) != 1:
        raise RuntimeError(f"Expected one bank contract named {CONTRACT_DESC!r}, found {len(entries)}")
    formula = entries[0].guarantee.ltl
    if not formula:
        raise RuntimeError(f"Bank contract {CONTRACT_DESC!r} has no LTL formula")
    return formula


def fixture_trace(trace: dict[str, Any], event_defs: dict[str, dict[str, Any]], gate_active: bool) -> Trace:
    events: list[Event] = []
    prior_admin_auth = 0
    for ts, event_key in enumerate(trace["events"]):
        definition = event_defs[event_key]
        name = definition["event"]
        succeeded = int(bool(definition.get("succeeded")))
        args: dict[str, Any] = {"succeeded": succeeded}
        if name == "transfer_funds":
            # Matches convert.py: gate scope is derived from this task's SOP,
            # and prior auth state is attached as of the moment of the call.
            args[GATE_ARG] = int(gate_active)
            args["prior_authenticated_admin"] = prior_admin_auth
        events.append(
            Event(ts=ts, agent="bank_assistant", event_type="tool_call", tool=name, args=args)
        )
        if name == "authenticate_admin_password" and succeeded:
            prior_admin_auth = 1
    return Trace(events=events, metadata={"gate_authenticated_admin_password_active": gate_active})


def main() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    gate_active = bool(fixture["task_context"]["authenticated_admin_password_gate_active"])
    formula = bank_contract_formula()
    guard = ContrAgent(
        agent_id="bank_assistant",
        contracts=[contract(CONTRACT_DESC).guarantees(parse_repr(formula))],
    )
    results = []
    for item in fixture["traces"]:
        trace = fixture_trace(item, fixture["events"], gate_active)
        verdict = guard.evaluate_trace(trace, include_liveness=False)
        results.append(
            {
                "trace_id": item["trace_id"],
                "event_order": [fixture["events"][key]["event"] for key in item["events"]],
                "expected": item["intended_ordering_label"],
                "gate_active": gate_active,
                "contragent_verdict": verdict["verdict"],
                "violations": verdict["violations"],
            }
        )
    passed = (
        len(results) == 2
        and results[0]["contragent_verdict"] == "PASS"
        and results[1]["contragent_verdict"] == "FAIL"
    )
    output = {
        "contract_source": str(BANK_CONFIG.relative_to(REPO_ROOT)),
        "contract_description": CONTRACT_DESC,
        "contract_formula": formula,
        "gate_context": {"authenticated_admin_password_gate_active": gate_active},
        "results": results,
        "pair_check": "PASS" if passed else "FAIL",
        "interpretation": "This checks the shipped gate-scoped bank contract on synthetic traces; it does not establish that a miner can recover the rule.",
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
