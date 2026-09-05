"""Shared helpers for comprehensive pattern / atom tests."""

from __future__ import annotations

import contragent


def make_guard(*contracts) -> contragent.ContrAgent:
    """Build a quiet ``ContrAgent`` guard from a list of contracts.

    Each contract may be a plain ``DetFormula`` (wrapped into the
    canonical ``{"guarantee": det}`` dict expected by ``BaseGuard``)
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
