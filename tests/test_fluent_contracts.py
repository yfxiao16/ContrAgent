"""Contracts written with the fluent helper behave like YAML ones: their
description is what a refusal reports, and their eventualities are
decided at session end."""

from __future__ import annotations

from contragent import ContrAgent, contract, parse_repr

PRECEDENCE = (
    "(!(called('transfer_funds')) U called('verify_identity')) | G(!(called('transfer_funds')))"
)
RECEIPT = "G(called('transfer_funds') -> F(called('send_receipt')))"


def test_refusal_cites_the_contract_description():
    guard = ContrAgent(
        agent_id="a",
        contracts=[
            contract("identity must be verified before funds move").guarantees(
                parse_repr(PRECEDENCE)
            )
        ],
    )
    result = guard.guard_before("transfer_funds", {"amount": 5})
    assert result.blocked
    assert "identity must be verified before funds move" in result.feedback
    assert "U called" not in result.feedback  # the formula is not echoed


def test_yaml_contract_description_reaches_the_refusal(tmp_path):
    cfg = tmp_path / "bank.yaml"
    cfg.write_text(
        "version: '1'\nagents:\n  '*':\n    contracts:\n"
        "      - desc: 'identity must be verified before funds move'\n"
        f'        G: {{ltl: "{PRECEDENCE}"}}\n'
    )
    guard = ContrAgent(agent_id="a", config=cfg)
    result = guard.guard_before("transfer_funds", {"amount": 5})
    assert result.blocked and "identity must be verified before funds move" in result.feedback


def test_fluent_eventuality_is_decided_at_session_end():
    guard = ContrAgent(
        agent_id="a",
        contracts=[
            contract("every transfer is eventually receipted").guarantees(parse_repr(RECEIPT))
        ],
    )
    guard.guard_before("transfer_funds", {})
    guard.guard_after("transfer_funds", {"ok": True})
    pending = guard.finish_session()
    assert [v.desc for v in pending] == ["every transfer is eventually receipted"]


def test_fluent_eventuality_discharged_is_clean():
    guard = ContrAgent(
        agent_id="a",
        contracts=[
            contract("every transfer is eventually receipted").guarantees(parse_repr(RECEIPT))
        ],
    )
    guard.guard_before("transfer_funds", {})
    guard.guard_after("transfer_funds", {"ok": True})
    guard.guard_before("send_receipt", {})
    guard.guard_after("send_receipt", {"ok": True})
    assert guard.finish_session() == []


def test_contract_without_desc_is_reported_in_words(tmp_path):
    """No desc: the refusal carries the formula read back in words, never the
    formula itself."""
    cfg = tmp_path / "bank.yaml"
    cfg.write_text(
        f"version: '1'\nagents:\n  '*':\n    contracts:\n      - G: {{ltl: \"{PRECEDENCE}\"}}\n"
    )
    guard = ContrAgent(agent_id="a", config=cfg)
    result = guard.guard_before("transfer_funds", {"amount": 5})
    assert result.blocked
    assert "must precede" in result.feedback
    assert "U called" not in result.feedback

    raw = ContrAgent(agent_id="a", contracts=[parse_repr(PRECEDENCE)])
    assert "must precede" in raw.guard_before("transfer_funds", {"amount": 5}).feedback
