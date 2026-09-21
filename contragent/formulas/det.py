"""Deterministic contract formulas.

A :class:`DetFormula` wraps an ALTL\\ :sub:`f` formula over interaction
predicates together with a human-readable description. Contracts carry
one such formula for the assumption and one for the guarantee; the
supervisor compiles each to a DFA monitor and evaluates it pointwise
over the tool-call trace.

The module also holds the tool-name conventions shared by the grounding
layer and the formula constructors: ``"bash:rm -rf"`` denotes the
physical tool ``bash`` restricted to calls whose arguments match the
pattern ``rm -rf`` (a :math:`\\mathsf{ArgHas}` refinement of
:math:`\\mathsf{Call}`), while ``"server:tool"`` is a namespaced tool
name and is left intact. Predicate keys write the tool name in its
canonical spelling (:mod:`contragent.formulas.tool_names`), so a
contract keys on one spelling however its author typed the name.
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass
from typing import Any

from contragent.formulas.formula import Atom, Formula, Var

_NAMESPACED_TOOL_RE = _re.compile(r"^[A-Za-z_][\w-]*:[A-Za-z_][\w-]*$")


def is_namespaced_tool_name(tool: str) -> bool:
    """``server:tool`` style names (no spaces) are namespaces, not refinements."""
    return bool(_NAMESPACED_TOOL_RE.match(str(tool)))


def physical_tool(tool: str) -> str:
    """Strip an argument refinement, ``"bash:rm -rf"`` -> ``"bash"``."""
    if ":" in tool and not is_namespaced_tool_name(tool):
        return tool.split(":", 1)[0]
    return tool


def called(tool: str) -> Atom:
    """The :math:`\\mathsf{Call}` predicate for ``tool``, refined by argument pattern if given."""
    tool = str(tool)
    if ":" in tool and not is_namespaced_tool_name(tool):
        physical, pattern = tool.split(":", 1)
        return Atom("called_with", physical, pattern)
    return Atom("called", tool)


def count_var(tool: str) -> Var:
    """The call counter of ``tool`` (the :math:`\\mathsf{Count}` quantity)."""
    tool = str(tool)
    if ":" in tool and not is_namespaced_tool_name(tool):
        physical, pattern = tool.split(":", 1)
        return Var("count_with", physical, pattern)
    return Var("count", tool)


# Internal aliases kept for the grounding and generation modules.
_is_namespaced_tool_name = is_namespaced_tool_name
_physical_tool = physical_tool
_called = called
_count_var = count_var


@dataclass(frozen=True)
class DetFormula:
    """An ALTL\\ :sub:`f` formula with its description.

    Attributes:
        formula: The formula AST (see :mod:`contragent.formulas.formula`).
        desc: Human-readable statement of the property.
        kind: Where the formula came from: ``"ltl"`` for a formula written
            directly, ``"ir:<shape>"`` for one produced by the formulation
            pipeline, ``"assumption"`` for a compiled assumption.
        liveness: True when the formula has an unbounded eventuality
            (``F``, ``U``) that cannot be refuted by a finite prefix. The
            supervisor evaluates such formulas only at session end.
        args: Constructor arguments, kept for round-tripping.
        enforcement_strategy: Optional strategy overriding the default
            block when this formula is violated.
    """

    formula: Formula
    desc: str
    kind: str = "ltl"
    liveness: bool = False
    args: tuple = ()
    enforcement_strategy: Any = None

    def __rshift__(self, other):
        return self.formula >> other

    def __and__(self, other):
        return self.formula & other

    def __or__(self, other):
        return self.formula | other

    def __invert__(self):
        return ~self.formula


__all__ = [
    "DetFormula",
    "called",
    "count_var",
    "physical_tool",
    "is_namespaced_tool_name",
]
