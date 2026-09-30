"""``ContrAgent.wrap``: both hooks around a tool, in every calling shape."""

from __future__ import annotations

import pytest

from contragent import ContractViolation, ContrAgent, contract, parse_repr

PRECEDENCE = (
    "(!(called('transfer_funds')) U called('verify_identity')) | G(!(called('transfer_funds')))"
)
NO_EGRESS = "G((called('read_file') -> G(!(called('send_email')))))"
NO_KEY = "G(!(output_has('read_file', 'BEGIN PRIVATE KEY')))"


def _guard():
    return ContrAgent(
        agent_id="a",
        contracts=[
            contract("identity must be verified before funds move").guarantees(
                parse_repr(PRECEDENCE)
            ),
            contract("no email after a file read").guarantees(parse_repr(NO_EGRESS)),
            contract("file reads carry no private key")
            .assume(parse_repr(NO_KEY))
            .guarantees(parse_repr("G(!(called('nothing')))")),
        ],
    )


def test_decorator_refuses_then_allows():
    guard = _guard()
    ran = []

    @guard.wrap
    def verify_identity(user_id: str) -> dict:
        ran.append("verify")
        return {"ok": True}

    @guard.wrap
    def transfer_funds(amount: float, to: str) -> dict:
        ran.append("transfer")
        return {"ok": True, "amount": amount}

    refusal = transfer_funds(500, "ACME")
    assert isinstance(refusal, str) and "identity must be verified" in refusal
    assert ran == []  # the tool did not run
    verify_identity("u1")
    assert transfer_funds(500, to="ACME") == {"ok": True, "amount": 500}
    assert ran == ["verify", "transfer"]


def test_mapping_and_list_keep_their_shape():
    guard = _guard()

    def read_file(path: str) -> str:
        return "notes"

    def send_email(to: str, body: str) -> dict:
        return {"sent": True}

    tools = guard.wrap({"read_file": read_file, "send_email": send_email})
    assert set(tools) == {"read_file", "send_email"}
    assert tools["send_email"]("x", "hi") == {"sent": True}  # nothing read yet
    tools["read_file"]("notes.txt")
    assert "no email after a file read" in tools["send_email"]("x", "hi")

    wrapped = guard.wrap([read_file, send_email])
    assert [f.contragent_tool for f in wrapped] == ["read_file", "send_email"]


def test_withheld_result_is_replaced():
    guard = _guard()

    @guard.wrap
    def read_file(path: str) -> str:
        return "-----BEGIN PRIVATE KEY-----"

    shown = read_file("id_rsa")
    assert "withheld" in shown and "PRIVATE KEY" not in shown.split("withheld")[0]


def test_raise_mode():
    guard = _guard()

    @guard.wrap(on_block="raise")
    def transfer_funds(amount: float, to: str) -> dict:
        return {"ok": True}

    with pytest.raises(ContractViolation) as exc:
        transfer_funds(1.0, "x")
    assert "identity must be verified" in exc.value.feedback
    assert exc.value.result.blocked


def test_explicit_name_and_bad_mode():
    guard = _guard()
    wrapped = guard.wrap(lambda amount, to: {"ok": True}, name="transfer_funds")
    assert "identity must be verified" in wrapped(1.0, "x")
    with pytest.raises(ValueError):
        guard.wrap(lambda: None, on_block="explode")
