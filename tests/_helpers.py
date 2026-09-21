"""Shared helpers for comprehensive pattern / atom tests."""

from __future__ import annotations

import contragent
from contragent.formulas.det import DetFormula
from contragent.formulas.parser import parse_repr


def ltl(
    text: str,
    desc: str = "",
    *,
    liveness: bool = False,
    enforcement_strategy=None,
) -> DetFormula:
    """Build a ``DetFormula`` from an ALTL_f string.

    This is the path every shipped contract takes (see
    ``contragent.config._compile_ltl``): the ``ltl:`` string of a YAML
    entry is parsed with ``parse_repr`` and wrapped as ``kind="ltl"``.
    Tests write their formulas the same way so that the suite exercises
    the parser and, where a shipped contract has the right shape, the
    published string itself.
    """
    return DetFormula(
        formula=parse_repr(text),
        desc=desc or text,
        kind="ltl",
        liveness=liveness,
        enforcement_strategy=enforcement_strategy,
    )


def make_guard(*contracts) -> contragent.ContrAgent:
    """Build a quiet ``ContrAgent`` guard from a list of contracts.

    Each contract may be a plain ``DetFormula`` (wrapped into the
    canonical ``{"guarantee": det}`` dict expected by ``ContrAgent``)
    or a pre-built dict / NL string. Banners + auto-summary are
    suppressed so the test output stays clean.
    """
    wrapped = []
    for c in contracts:
        if hasattr(c, "formula"):
            wrapped.append({"guarantee": c})
        else:
            wrapped.append(c)
    return contragent.ContrAgent(
        contracts=wrapped,
    )


def violation_text(g: contragent.ContrAgent) -> str:
    """Flatten ``guard.violations`` into a single string for substring asserts."""
    return " ".join(str(v) for v in g.violations)
