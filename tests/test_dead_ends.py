"""The static dead-end check: the two decidable patterns are reported,
and a library without them passes (subject to the unchecked conditions).
This is the check whose verdicts on the shipped libraries the results
ledger reports."""

from __future__ import annotations

from pathlib import Path

from contragent.analysis.dead_ends import check_dead_ends
from contragent.config import load_system
from contragent.formulas.parser import parse_repr

ROOT = Path(__file__).resolve().parent.parent


def _library(*formulas: str):
    return [parse_repr(f) for f in formulas]


def test_rate_limited_obligation_is_a_dead_end():
    report = check_dead_ends(
        _library(
            "G(called('detect_fraud') -> X(called('freeze_account')))",
            "G(Var('count', 'freeze_account') <= 1)",
        )
    )
    assert not report.ok
    assert [o.tool for o in report.rate_limited] == ["freeze_account"]
    assert report.bounded_tools == {"freeze_account": 1}
    assert "freeze_account" in report.summary()


def test_competing_next_obligations_are_a_dead_end():
    report = check_dead_ends(
        _library(
            "G(called('detect_fraud') -> X(called('freeze_account')))",
            "G(called('detect_fraud') -> X(called('notify_customer')))",
        )
    )
    assert not report.ok
    assert len(report.competing) == 1


def test_two_eventually_obligations_can_both_be_discharged():
    report = check_dead_ends(
        _library(
            "G(called('detect_fraud') -> F(called('freeze_account')))",
            "G(called('detect_fraud') -> F(called('notify_customer')))",
        )
    )
    assert report.ok
    assert len(report.obligations) == 2 and not report.competing


def test_shipped_sopbench_library_passes():
    system = load_system(ROOT / "contragent" / "contracts" / "sopbench" / "bank.yaml")
    report = check_dead_ends(system.contracts)
    assert report.ok
    assert report.unchecked  # the semantic conditions are reported, not decided
