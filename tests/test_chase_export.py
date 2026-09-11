"""Tests for the CHASE logics export."""

from __future__ import annotations

import re

import pytest

from contragent.analysis.chase import (
    FINITE_TRACE_CONTRACT,
    MAX_CONTRACT_IDENT,
    ChaseExportError,
    contract_identifiers,
    export_library,
    export_logics,
    smv_declarations,
)
from contragent.config import bundled_libraries_root
from contragent.formulas.parser import parse_repr
from contragent.models.agent import Agent
from contragent.models.contract import Contract

# Tokens allowed inside a formula line by CHASE's LogicsContracts grammar.
_FORMULA_TOKEN = re.compile(
    r"\s*(?:/\\|\\/|->|<->|<=|>=|!=|[!()<>=]|\b[GFXUR]\b|[A-Za-z][A-Za-z0-9_]*|-?\d+(?:\.\d+)?|;)"
)
_DECL = re.compile(
    r"^(?:proposition [A-Za-z][A-Za-z0-9_]*;"
    r"|integer(?: \(\d+:\d+\))? variable [A-Za-z][A-Za-z0-9_]*;)"
)


def _tokens_ok(line: str) -> bool:
    pos = 0
    while pos < len(line):
        m = _FORMULA_TOKEN.match(line, pos)
        if not m:
            return False
        pos = m.end()
    return True


def _strip_comments(text: str) -> list[str]:
    return [ln.split("#", 1)[0].rstrip() for ln in text.splitlines()]


def _contracts(*pairs):
    bot = Agent(id="bot")
    out = []
    for i, (a, g) in enumerate(pairs):
        out.append(
            Contract(
                agent=bot,
                assumption=parse_repr(a) if a else None,
                guarantee=parse_repr(g),
                desc=f"rule {i}",
            )
        )
    return out


class TestGrammarShape:
    def test_header_declarations_then_contracts(self):
        text = export_logics(
            _contracts((None, "G((Var('count', 'pay_bill') <= 3))")),
            name="demo",
            semantics="infinite",
        )
        lines = [ln for ln in _strip_comments(text) if ln.strip()]
        assert lines[0] == "NAME: demo;"
        decls = [ln for ln in lines if ln.startswith(("proposition", "integer"))]
        assert decls and all(_DECL.match(ln) for ln in decls)
        first_contract = next(i for i, ln in enumerate(lines) if ln.startswith("CONTRACT"))
        assert all(not ln.startswith(("proposition", "integer")) for ln in lines[first_contract:])

    def test_formula_lines_use_only_grammar_tokens(self):
        text = export_library(bundled_libraries_root() / "sopbench" / "bank.yaml")
        for ln in _strip_comments(text):
            if ln.startswith("    "):
                assert _tokens_ok(ln.strip()), ln

    def test_every_identifier_in_formulas_is_declared(self):
        text = export_library(bundled_libraries_root() / "benchmark" / "rjudge.yaml")
        lines = _strip_comments(text)
        declared = set(
            re.findall(
                r"^(?:proposition|integer(?: \(\d+:\d+\))? variable) ([A-Za-z]\w*);",
                text,
                flags=re.M,
            )
        )
        used = set()
        for ln in lines:
            if ln.startswith("    "):
                used.update(
                    t
                    for t in re.findall(r"\b[A-Za-z]\w*\b", ln)
                    if t not in {"G", "F", "X", "U", "R", "true", "false"}
                )
        assert used <= declared, used - declared

    def test_contract_identifiers_are_unique(self):
        text = export_library(bundled_libraries_root() / "sopbench" / "hotel.yaml")
        ids = contract_identifiers(text)
        assert len(ids) == len(set(ids)) and ids

    def test_contract_identifiers_fit_the_chase_console(self):
        # CHASE's console segfaults in verify/refinement on names longer than eight characters.
        for path in bundled_libraries_root().rglob("*.yaml"):
            for ident in contract_identifiers(export_library(path)):
                assert len(ident) <= MAX_CONTRACT_IDENT, (path, ident)


class TestGrounding:
    def test_saturating_counter_gets_range_above_largest_bound(self):
        text = export_logics(
            _contracts(
                (None, "G((Var('count', 'send') <= 3))"), (None, "G((Var('count', 'send') <= 7))")
            ),
            semantics="infinite",
        )
        assert "integer (0:8) variable v_count_send;" in text

    def test_argument_quantities_are_unbounded_integers(self):
        text = export_logics(
            _contracts((None, "G((called('t') -> (ArgValue('t', 'amount') <= 42)))")),
            semantics="infinite",
        )
        assert "integer variable num_t_amount;" in text
        assert "(num_t_amount <= 42)" in text

    def test_predicates_become_propositions_named_by_key(self):
        text = export_logics(
            _contracts((None, "G((called('bash') -> !(arg_field_has('bash', 'cmd', 'rm -rf'))))")),
            semantics="infinite",
        )
        assert "proposition p_called_bash;" in text
        assert re.search(
            r"proposition p_arg_field_has_bash_cmd_rm_rf;\s+# arg_field_has\(bash, cmd, rm -rf\)",
            text,
        )


class TestSemantics:
    def test_finite_adds_alive_axiom_and_guards(self):
        text = export_logics(
            _contracts(
                (
                    "G(!(output_has('a', 'SECRET')))",
                    "G((called('a') -> F(called('b'))))",
                )
            )
        )
        assert f"CONTRACT {FINITE_TRACE_CONTRACT}:" in text
        assert "(alive /\\ (alive U (G (! alive))))" in text
        assert "(G (alive -> (p_called_a -> (F (alive /\\ p_called_b)))))" in text
        assert "  Assumptions:\n    (G (alive -> (! p_output_has_a_SECRET)));" in text

    def test_infinite_exports_formulas_verbatim(self):
        text = export_logics(
            _contracts((None, "(!(called('b')) U called('a')) | G(!(called('b')))")),
            semantics="infinite",
        )
        assert "alive" not in text
        assert "(((! p_called_b) U p_called_a) \\/ (G (! p_called_b)))" in text

    def test_rejects_unknown_semantics(self):
        with pytest.raises(ValueError):
            export_logics([], semantics="omega")


class TestSmvDeclarations:
    TEXT = (
        "NAME: t;\nproposition p_called_a;\n"
        "integer (0:4) variable v_count_a;\ninteger variable num_a_amount;\n"
    )

    def test_types_follow_the_declarations(self):
        assert smv_declarations(self.TEXT) == [
            "\tp_called_a : boolean;\n",
            "\tv_count_a : 0..4;\n",
            "\tnum_a_amount : integer;\n",
        ]

    def test_unbounded_integers_take_the_given_range(self):
        assert "\tnum_a_amount : 0..100;\n" in smv_declarations(self.TEXT, int_range=(0, 100))

    def test_every_declaration_of_an_export_is_typed(self):
        text = export_library(bundled_libraries_root() / "benchmark" / "tau2_bench.yaml")
        n_decl = len(re.findall(r"^(?:proposition|integer)", text, flags=re.M))
        assert len(smv_declarations(text)) == n_decl


def test_all_shipped_libraries_export():
    root = bundled_libraries_root()
    for path in sorted(root.rglob("*.yaml")):
        text = export_library(path)
        assert contract_identifiers(text), path


def test_export_error_on_unsupported_node():
    class Weird:  # not a formula node
        pass

    bot = Agent(id="bot")
    c = Contract(agent=bot, guarantee=Weird(), desc="odd")
    with pytest.raises(ChaseExportError):
        export_logics([c])
