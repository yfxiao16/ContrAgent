"""Tests for the pychase-based CHASE bridge (skipped without CHASE's bindings)."""

from __future__ import annotations

import importlib.util

import pytest

from contragent.analysis.chase import ChaseUnavailable, PychaseTranslator
from contragent.config import bundled_libraries_root, load_system

pychase_present = importlib.util.find_spec("pychase") is not None


def test_translator_reports_missing_bindings_clearly():
    if pychase_present:
        pytest.skip("pychase is installed")
    with pytest.raises(ChaseUnavailable):
        PychaseTranslator()


@pytest.mark.skipif(not pychase_present, reason="requires CHASE's pychase bindings")
class TestWithPychase:
    def test_contract_translates_with_declarations(self):
        contracts = load_system(bundled_libraries_root() / "benchmark" / "rjudge.yaml").contracts
        t = PychaseTranslator()
        c = t.contract(contracts[0], "C1")
        assert c.getName().getString() == "C1"
        assert len(list(c.declarations)) == len(t.variables)
        assert list(c.guarantees)

    def test_conjunction_over_a_library(self):
        contracts = load_system(bundled_libraries_root() / "sopbench" / "hotel.yaml").contracts[:4]
        t = PychaseTranslator()
        r = t.conjoin(contracts, name="hotel_head")
        assert r.getName().getString() == "hotel_head"
        assert len(list(r.declarations)) == len(t.variables)

    def test_shared_variables_are_identified(self):
        contracts = load_system(bundled_libraries_root() / "sopbench" / "hotel.yaml").contracts[:2]
        t = PychaseTranslator()
        a, b = (t.contract(c, f"C{i}") for i, c in enumerate(contracts, 1))
        corr = PychaseTranslator.shared_variables(a, b)
        assert corr and all(k == v for k, v in corr.items())
