"""A contract applies to every spelling of the tool it names.

Predicate keys are dictionary keys. Before canonicalisation a contract
written against ``cancel_pending_order`` was inert on a call that arrived
as ``Cancel_Pending_Order``, with a trailing space, or under the MCP wire
name ``mcp__retail__cancel_pending_order``: the ``called`` key never
matched, and the shipped shapes read that as satisfied. The strings below
are lifted verbatim from the shipped libraries.
"""

from __future__ import annotations

import pytest

from contragent.config import bundled_libraries_root
from contragent.core import ContrAgent
from contragent.formulas._pred_key import pred_key
from contragent.formulas.tool_names import canonical_tool, tool_aliases
from tests._helpers import ltl

# contragent/contracts/benchmark/tau2_bench.yaml
PRECEDENCE = (
    "((!(called('cancel_pending_order')) U called('get_order_details'))"
    " | G(!(called('cancel_pending_order'))))"
)
# contragent/contracts/benchmark/agentdojo.yaml
FORBIDDEN_FILE = "G((called('delete_file') -> !(arg_field_has('delete_file', 'file_id', '^13$'))))"

SPELLINGS = [
    "cancel_pending_order",
    "Cancel_Pending_Order",
    "cancel_pending_order ",
    "mcp__retail__cancel_pending_order",
]


def _guard(*texts: str) -> ContrAgent:
    return ContrAgent(agent_id="a", contracts=[ltl(t) for t in texts], mode="gate")


def test_strings_are_shipped():
    tau2 = (bundled_libraries_root() / "benchmark" / "tau2_bench.yaml").read_text()
    dojo = (bundled_libraries_root() / "benchmark" / "agentdojo.yaml").read_text()
    assert PRECEDENCE in tau2
    assert FORBIDDEN_FILE in dojo


class TestCanonicalForm:
    def test_pred_key_folds_case_and_whitespace_for_tool_keyed_predicates(self):
        assert pred_key("called", "Issue_Refund ") == "called(issue_refund)"
        assert pred_key("count", " Issue_Refund") == "count(issue_refund)"
        assert pred_key("arg_field_has", "Pay", "amount", "^0$") == pred_key(
            "arg_field_has", "pay", "amount", "^0$"
        )

    def test_pred_key_leaves_other_predicates_alone(self):
        assert pred_key("ctx", "Role", "Admin") == "ctx(Role, Admin)"
        assert pred_key("flow", "A", "B") == "flow(A, B)"

    def test_canonical_tool_keeps_mcp_prefix(self):
        assert canonical_tool(" Issue_Refund ") == "issue_refund"
        assert canonical_tool("mcp__finance__issue_refund") == "mcp__finance__issue_refund"

    def test_event_answers_to_bare_name_behind_mcp_prefix(self):
        aliases = tool_aliases("mcp__finance__Issue_Refund")
        assert aliases[0] == "mcp__finance__Issue_Refund"
        assert "Issue_Refund" in aliases
        assert "issue_refund" in aliases


class TestShippedPrecedence:
    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_every_spelling_is_blocked_before_the_lookup(self, spelling):
        guard = _guard(PRECEDENCE)
        result = guard.guard_before(spelling, {})
        assert result.blocked
        assert not result.allowed
        # The refused call left no trace.
        assert guard.trace.events == []

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_every_spelling_is_allowed_after_the_lookup(self, spelling):
        guard = _guard(PRECEDENCE)
        assert guard.guard_before("get_order_details", {}).allowed
        assert guard.guard_before(spelling, {}).allowed

    def test_lookup_under_any_spelling_satisfies_the_precedence(self):
        guard = _guard(PRECEDENCE)
        assert guard.guard_before("mcp__retail__Get_Order_Details", {}).allowed
        assert guard.guard_before("cancel_pending_order", {}).allowed


class TestCountersCountAliases:
    def test_count_bound_counts_the_mcp_spelling(self):
        guard = _guard("G((Var('count', 'issue_refund') <= 1))")
        assert guard.guard_before("issue_refund", {}).allowed
        result = guard.guard_before("mcp__finance__issue_refund", {})
        assert result.blocked

    def test_consecutive_count_runs_across_spellings(self):
        guard = _guard("G((Var('consecutive_count', 'search') <= 2))")
        assert guard.guard_before("search", {}).allowed
        assert guard.guard_before("Search", {}).allowed
        assert guard.guard_before("mcp__web__search", {}).blocked
        # A different tool resets the run.
        assert guard.guard_before("other", {}).allowed
        assert guard.guard_before("search", {}).allowed


class TestArgumentRulesFollowTheAlias:
    def test_forbidden_argument_is_caught_under_the_mcp_name(self):
        guard = _guard(FORBIDDEN_FILE)
        assert guard.guard_before("mcp__drive__delete_file", {"file_id": "13"}).blocked
        assert guard.guard_before("Delete_File", {"file_id": "12"}).allowed
