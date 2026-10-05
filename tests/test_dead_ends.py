"""The static dead-end check: the two decidable patterns are reported,
and a library without them passes (subject to the unchecked conditions).
This is the check whose verdicts on the shipped libraries the results
ledger reports."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from contragent.analysis.dead_ends import check_dead_ends
from contragent.config import load_system
from contragent.formulas.parser import parse_repr

ROOT = Path(__file__).resolve().parent.parent


def _library(*formulas: str):
    # The check reads ``guarantee``/``desc`` (ContrAgent contracts) or
    # ``formula``/``name`` (control-layer contracts); a bare formula is skipped.
    return [SimpleNamespace(formula=parse_repr(f), name=f"c{i}") for i, f in enumerate(formulas)]


def _yaml_library(tmp_path: Path, *guarantees: str | list[str]):
    contracts = []
    for i, guarantee in enumerate(guarantees):
        field = (
            [{"ltl": formula} for formula in guarantee]
            if isinstance(guarantee, list)
            else {"ltl": guarantee}
        )
        contracts.append({"desc": f"c{i}", "G": field})
    path = tmp_path / "contracts.yaml"
    path.write_text(
        yaml.safe_dump({"version": "1", "agents": {"*": {"contracts": contracts}}}),
        encoding="utf-8",
    )
    return load_system(path).contracts


@pytest.mark.parametrize("operator", ["X", "F"])
@pytest.mark.parametrize("bound", ["<= 1", "< 2"])
def test_yaml_rate_limited_obligation_is_a_dead_end(tmp_path, operator, bound):
    contracts = _yaml_library(
        tmp_path,
        f"G(called('approve') -> {operator}(called('notify')))",
        f"G(Var('count', 'notify') {bound})",
    )
    report = check_dead_ends(contracts)
    assert not report.ok
    assert [(o.label, o.trigger, o.tool, o.operator) for o in report.rate_limited] == [
        ("c0", "approve", "notify", operator)
    ]
    assert len(report.obligations) == 1
    assert report.bounded_tools == {"notify": 1}


def test_yaml_conjoined_obligation_and_bound_is_a_dead_end(tmp_path):
    contracts = _yaml_library(
        tmp_path,
        "G(called('approve') -> X(called('notify'))) & G(Var('count', 'notify') <= 1)",
    )
    report = check_dead_ends(contracts)
    assert not report.ok
    assert [o.tool for o in report.rate_limited] == ["notify"]
    assert report.bounded_tools == {"notify": 1}


@pytest.mark.parametrize("as_list", [False, True])
def test_yaml_competing_next_obligations_are_a_dead_end(tmp_path, as_list):
    formulas = [
        "G(called('approve') -> X(called('notify')))",
        "G(called('approve') -> X(called('archive')))",
    ]
    contracts = _yaml_library(tmp_path, formulas) if as_list else _yaml_library(tmp_path, *formulas)
    report = check_dead_ends(contracts)
    assert not report.ok
    assert len(report.obligations) == 2
    assert [(first.tool, second.tool) for first, second in report.competing] == [
        ("notify", "archive")
    ]


def test_yaml_eventually_obligations_in_a_list_are_not_competing(tmp_path):
    contracts = _yaml_library(
        tmp_path,
        [
            "G(called('approve') -> F(called('notify')))",
            "G(called('approve') -> F(called('archive')))",
        ],
    )
    report = check_dead_ends(contracts)
    assert report.ok
    assert [(o.label, o.tool, o.operator) for o in report.obligations] == [
        ("c0", "notify", "F"),
        ("c0", "archive", "F"),
    ]
    assert not report.competing


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
    assert "rate-limited" in report.summary()


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


def test_shipped_tau2_library_obligations_are_checked():
    system = load_system(ROOT / "contragent" / "contracts" / "benchmark" / "tau2_bench.yaml")
    report = check_dead_ends(system.contracts)
    # The ten called-trigger obligations have no bound on their required tools.
    assert len(report.obligations) == 10
    assert report.bounded_tools["cancel_pending_order"] == 2
    assert report.ok
    assert report.unchecked
