"""``verbose=True``: the session prints as a timeline, plain text off a tty."""

from __future__ import annotations

import io

from contragent import ContrAgent
from contragent.console import Console


def test_verbose_timeline(capsys):
    guard = ContrAgent(agent_id="bank_agent", config="examples/bank.yaml", verbose=True)
    guard.guard_before("transfer_funds", {"amount": 500, "to": "ACME"})
    guard.guard_before("verify_identity", {"user_id": "u1"})
    guard.guard_after("verify_identity", {"ok": True})
    guard.guard_before("read_file", {"path": "id_rsa"})
    guard.guard_after("read_file", "-----BEGIN PRIVATE KEY-----")
    guard.guard_before("transfer_funds", {"amount": 500, "to": "ACME"})
    guard.guard_after("transfer_funds", {"ok": True})
    guard.finish_session()
    guard.finish_session()  # idempotent: the summary prints once
    out = capsys.readouterr().out
    assert "\x1b[" not in out  # not a tty: no colour
    assert out.startswith("bank.yaml · 5 contracts · agent bank_agent · mode gate\n│\n")
    assert '⊘ transfer_funds(amount=500, to="ACME")\n│   identity must be verified' in out
    assert '● verify_identity(user_id="u1")' in out
    assert "◐ result withheld · no private key" in out
    assert out.count("■ session end") == 1
    assert "1 obligation pending" in out and "◌ every transfer is eventually receipted" in out


def test_clean_session_and_color_flag():
    buf = io.StringIO()
    console = Console(buf, color=True)
    console.banner(library="x.yaml", contracts=1, agent_id="a", mode="gate", version="1")
    console.summary(calls=2, refused=0, withheld=0, pending=[])
    text = buf.getvalue()
    assert "\x1b[" in text and "clean" in text
