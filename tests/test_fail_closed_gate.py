"""A call the contracts cannot be evaluated on is refused, not passed.

The paper defines an interaction predicate as a total function into
{true, false}. When a call arrives without the arguments a contract
reads, or with an amount a numeric predicate cannot read as a number,
the implementation has no value to give. Reading the predicate as false
would satisfy every shipped guarantee shape, so ``guard_before`` refuses
the call instead. The strings below are lifted verbatim from the shipped
libraries; the ``ArgValue`` cap goes through the same loader as a YAML
``ltl:`` entry.
"""

from __future__ import annotations

import pytest

from contragent.config import ConstraintEntry, _compile_ltl, bundled_libraries_root
from contragent.core import ContrAgent
from contragent.formulas._compare import to_number
from contragent.models.trace import Event, Trace
from contragent.tracer.grounding import (
    collect_content_atoms,
    ground,
    grounding_misses,
    reset_grounding_misses,
)
from tests._helpers import ltl

# contragent/contracts/benchmark/agentdojo.yaml
FORBIDDEN_FILE = "G((called('delete_file') -> !(arg_field_has('delete_file', 'file_id', '^13$'))))"
# contragent/contracts/benchmark/tau2_bench.yaml
GB_RANGE = (
    "G(((Var('arg_numeric', 'refuel_data', 'gb_amount') >= 0)"
    " & (Var('arg_numeric', 'refuel_data', 'gb_amount') <= 2)))"
)
# The Cat-C spelling of tau2_bench.yaml: a term comparison, loaded by the
# infix fallback of _compile_ltl exactly as a YAML entry would be.
CAP = "G(Implies(called(pay), Not(Gt(ArgValue(pay, amount), Const(1000)))))"
GB_CAP = "G(Implies(called(refuel_data), Le(ArgValue(refuel_data, gb_amount), Const(2))))"


def _guard(*texts: str, mode: str = "gate") -> ContrAgent:
    formulas = [_compile_ltl(ConstraintEntry(ltl=t)) for t in texts]
    return ContrAgent(agent_id="a", contracts=formulas, mode=mode)


def test_strings_are_shipped():
    tau2 = (bundled_libraries_root() / "benchmark" / "tau2_bench.yaml").read_text()
    dojo = (bundled_libraries_root() / "benchmark" / "agentdojo.yaml").read_text()
    assert FORBIDDEN_FILE in dojo
    assert GB_RANGE in tau2


class TestMissingArguments:
    def test_forbidden_argument_is_blocked(self):
        result = _guard(FORBIDDEN_FILE).guard_before("delete_file", {"file_id": "13"})
        assert result.blocked and not result.allowed

    @pytest.mark.parametrize("args", [{}, None])
    def test_a_call_without_arguments_is_refused(self, args):
        guard = _guard(FORBIDDEN_FILE)
        result = guard.guard_before("delete_file", args)
        assert result.blocked
        assert not result.allowed
        assert result.violations[0].rule_id == "args:unevaluable"
        assert "arguments" in result.feedback
        assert guard.trace.events == []
        assert guard.violations[-1]["action"] == "BLOCKED"

    def test_the_mcp_spelling_is_refused_too(self):
        assert _guard(FORBIDDEN_FILE).guard_before("mcp__drive__delete_file", None).blocked

    def test_an_absent_pattern_field_is_false_not_a_gap(self):
        # No file_id means no file_id matches ^13$: the guarantee holds.
        assert _guard(FORBIDDEN_FILE).guard_before("delete_file", {"other": 1}).allowed

    def test_a_tool_no_contract_reads_the_arguments_of_is_unaffected(self):
        guard = _guard(FORBIDDEN_FILE)
        assert guard.guard_before("read_file", None).allowed
        assert guard.guard_before("read_file", {}).allowed

    def test_flag_mode_records_the_refusal_without_gating(self):
        result = _guard(FORBIDDEN_FILE, mode="flag").guard_before("delete_file", None)
        assert result.allowed
        assert [v.action for v in result.violations] == ["observed"]

    def test_escape_hatch_restores_the_permissive_reading(self, monkeypatch):
        monkeypatch.setenv("CONTRAGENT_ALLOW_MISSING_ARGS", "1")
        assert _guard(FORBIDDEN_FILE).guard_before("delete_file", None).allowed


class TestNumericPredicates:
    @pytest.mark.parametrize("amount", [5000, "5000", "$5,000", "5,000", "5000 USD", "1e400"])
    def test_every_spelling_over_the_cap_is_blocked(self, amount):
        result = _guard(CAP).guard_before("pay", {"amount": amount})
        assert result.blocked
        assert result.violations[0].rule_id != "args:unevaluable"

    @pytest.mark.parametrize("amount", [500, "500", "$999.99", "1,000"])
    def test_a_value_under_the_cap_is_allowed(self, amount):
        assert _guard(CAP).guard_before("pay", {"amount": amount}).allowed

    @pytest.mark.parametrize("amount", ["5,50", "abc", ""])
    def test_a_value_that_is_not_a_number_is_refused_not_passed(self, amount):
        result = _guard(CAP).guard_before("pay", {"amount": amount})
        assert result.blocked
        assert result.violations[0].rule_id == "args:unevaluable"

    def test_the_field_the_cap_reads_must_be_present(self):
        result = _guard(CAP).guard_before("pay", {"currency": "USD"})
        assert result.blocked
        assert result.violations[0].rule_id == "args:unevaluable"

    @pytest.mark.parametrize(
        "gb, blocked",
        [("1.5", False), ("2", False), ("2 GB", False), ("3", True), ("$3", True), ("2,500", True)],
    )
    def test_arg_numeric_and_argvalue_paths_agree(self, gb, blocked):
        for text in (GB_RANGE, GB_CAP):
            result = _guard(text).guard_before("refuel_data", {"gb_amount": gb})
            assert result.blocked is blocked, (text, gb)

    def test_five_comma_fifty_is_not_read_as_five_hundred_fifty(self):
        assert to_number("5,50") is None
        assert to_number("5,500") == 5500

    def test_coercion_table(self):
        assert to_number("5000") == 5000
        assert to_number("$5,000") == 5000
        assert to_number("5000 USD") == 5000
        assert to_number("12 %") == 12
        assert to_number("1e400") == float("inf")
        assert to_number("USD") is None
        assert to_number(None) is None
        assert to_number(True) is None
        assert to_number(3.5) == 3.5


class TestOfflineReplayCountsMisses:
    def test_a_numeric_field_with_no_number_is_counted_and_warned_once(self):
        reset_grounding_misses()
        formulas = [ltl(GB_RANGE)]
        atoms = collect_content_atoms(formulas)
        events = [
            Event(
                ts=0,
                agent="a",
                event_type="tool_call",
                tool="refuel_data",
                args={"gb_amount": None},
            ),
            Event(
                ts=1,
                agent="a",
                event_type="tool_call",
                tool="refuel_data",
                args={"gb_amount": None},
            ),
            Event(
                ts=2, agent="a", event_type="tool_call", tool="refuel_data", args={"gb_amount": "1"}
            ),
        ]
        with pytest.warns(UserWarning, match="could not be evaluated") as record:
            valuations = ground(Trace(events=events), content_atoms=atoms)
        assert len([w for w in record if "could not be evaluated" in str(w.message)]) == 1
        assert grounding_misses() == {("arg_numeric", "refuel_data", "gb_amount", "not_numeric"): 2}
        assert valuations[2]["arg_numeric(refuel_data, gb_amount)"] == 1
        reset_grounding_misses()
