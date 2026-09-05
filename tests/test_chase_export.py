"""Tests for the CHASE logics export."""

from __future__ import annotations

import re

import pytest

from contragent.analysis.chase import (
    ChaseExportError,
    contract_identifiers,
    export_library,
    export_logics,
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
        text = export_logics(_contracts(("F(called('a'))", "G((called('a') -> F(called('b'))))")))
        assert "CONTRACT finite_trace:" in text
        assert "(alive /\\ (alive U (G (! alive))))" in text
        assert "(G (alive -> (p_called_a -> (F (alive /\\ p_called_b)))))" in text
        assert "  Assumptions:\n    (F (alive /\\ p_called_a));" in text

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
