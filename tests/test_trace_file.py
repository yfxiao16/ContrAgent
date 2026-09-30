"""``trace_path``: the session is written as a replayable trace with the
decisions in its metadata."""

from __future__ import annotations

import json

from contragent import ContrAgent, Trace


def _session(**kw):
    guard = ContrAgent(agent_id="bank_agent", config="examples/bank.yaml", **kw)
    guard.guard_before("transfer_funds", {"amount": 500, "to": "ACME"})  # refused
    guard.guard_before("verify_identity", {"user_id": "u1"})
    guard.guard_after("verify_identity", {"ok": True})
    guard.guard_before("transfer_funds", {"amount": 500, "to": "ACME"})
    guard.guard_after("transfer_funds", {"ok": True})
    guard.guard_before("read_file", {"path": "id_rsa"})
    guard.guard_after("read_file", "-----BEGIN PRIVATE KEY-----")  # withheld
    return guard


def test_trace_file_is_written_at_session_end_and_replays(tmp_path):
    path = tmp_path / "session.json"
    guard = _session(trace_path=path)
    assert not path.exists()
    guard.finish_session()
    data = json.loads(path.read_text())
    tools = [e["tool"] for e in data["events"]]
    assert tools == ["verify_identity", "transfer_funds", "read_file"]  # the refused call is absent
    meta = data["metadata"]
    assert meta["agent"] == "bank_agent" and meta["library"] == "bank.yaml"
    assert [d["decision"] for d in meta["decisions"]] == ["refused", "withheld"]
    assert meta["decisions"][0]["contracts"] == ["identity must be verified before funds move"]
    assert meta["pending_obligations"] == ["every transfer is eventually receipted"]

    # The committed events replay clean under the same library: nothing
    # that happened violated a contract, and the receipt is still owed.
    replay = ContrAgent(agent_id="bank_agent", config="examples/bank.yaml")
    report = replay.evaluate_trace(Trace.load(path))
    assert [v["contract"] for v in report["violations"]] == [
        "every transfer is eventually receipted"
    ]


def test_save_trace_on_demand(tmp_path):
    guard = _session()
    out = guard.save_trace(tmp_path / "mid.json")
    data = json.loads(open(out).read())
    assert len(data["events"]) == 3 and data["metadata"]["pending_obligations"] == []
