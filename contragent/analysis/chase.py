"""Bridge to CHASE, the contract-based requirement-engineering framework.

CHASE (Nuzzo, Lora, Feldman, and Sangiovanni-Vincentelli, DATE 2018;
https://chase-cps.github.io) represents assume-guarantee contracts over
temporal-logic formulas and offers a contract algebra (composition,
conjunction, refinement) with model-checking and synthesis back ends
(NuSMV, slugs, gr1c). ContrAgent's contract libraries are ALTL\\ :sub:`f`
contracts over interaction predicates; this module maps them onto CHASE's
*logics* specification language so that a library can be analysed with
CHASE's design-time tools, and drives the CHASE console when its Python
bindings are installed.

Grounding
---------
CHASE's logics language is propositional with typed variables: atoms are
propositions or relations over integer/real/boolean variables. ContrAgent's
predicates carry parameters, so a library is *grounded* at export:

* every distinct instantiated predicate (``called(transfer)``,
  ``arg_field_has(bash, command, rm -rf)``, ...) becomes one proposition,
  named by its canonical key (:func:`contragent.formulas._pred_key.pred_key`);
* every quantity (``count(T)``, ``arg_numeric(T, f)``, ``ArgLength(T, f)``,
  ``CtxValue(k)``) becomes one integer variable; a saturating counter is
  declared with the range ``(0:K)`` where ``K`` exceeds every constant it is
  compared with, the finite expansion the ContrAgent paper uses;
* comparisons are emitted as relations inside the formulas.

The library is finite, so the grounding is finite and exact.

Finite traces
-------------
CHASE's back ends reason over infinite words while ContrAgent's monitors
evaluate LTL\\ :sub:`f` on finite traces. With ``semantics="finite"`` (the
default) the export applies the standard LTL\\ :sub:`f`-to-LTL translation
with a fresh proposition ``alive`` (De Giacomo and Vardi, 2013):
``X`` becomes the weak ``X(alive -> .)`` used by the runtime, ``U``
becomes ``. U (alive /\\ .)``, ``F`` becomes ``F(alive /\\ .)``, ``G``
becomes ``G(alive -> .)``, and a contract ``finite`` asserts
``alive /\\ (alive U G(!alive))``. With ``semantics="infinite"`` the
formulas are exported unchanged.

Usage
-----
::

    from contragent.analysis.chase import export_logics
    from contragent.config import load_system

    system = load_system("contragent/contracts/sopbench/bank.yaml")
    text = export_logics(system.contracts, name="bank")
    open("bank.logics", "w").write(text)

    # with CHASE's Python bindings (pychase, pychase_logicsLang) installed:
    from contragent.analysis.chase import ChaseSession
    s = ChaseSession("bank.logics")
    s.verify("c1", "bank_c1.smv")   # NuSMV model with its VAR block filled in

or ``contragent export-chase --config LIB -o LIB.logics``.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contragent.formulas.formula import (
    And,
    ArgLength,
    ArgValue,
    Atom,
    Const,
    CtxValue,
    Eq,
    F,
    G,
    Ge,
    Gt,
    Implies,
    Le,
    Lt,
    Not,
    Or,
    Subset,
    U,
    UnaryFn,
    Var,
    X,
)

_SATURATING = frozenset(
    {"count", "count_with", "token_count", "consecutive_count", "delegation_depth", "time_since"}
)
_IDENT_RE = re.compile(r"[^A-Za-z0-9_]+")


class ChaseExportError(ValueError):
    """A formula uses a construct that has no CHASE counterpart."""


class ChaseUnavailable(RuntimeError):
    """CHASE's Python bindings (``pychase_logicsLang``) are not installed."""


# CHASE's console (verify, refinement, synthesize) crashes on contract names
# longer than eight characters, so exported contracts are numbered and the
# description travels in the comment above each block.
MAX_CONTRACT_IDENT = 8
FINITE_TRACE_CONTRACT = "finite"


def _comment(text: str) -> str:
    """CHASE line comments run to the end of the line and may not contain backslashes."""
    return str(text).replace("\\", "").replace("\r", " ").replace("\n", " ")


def _ident(prefix: str, *parts: Any) -> str:
    raw = "_".join(str(p) for p in parts)
    raw = _IDENT_RE.sub("_", raw).strip("_")
    name = f"{prefix}_{raw}" if raw else prefix
    return name if name[0].isalpha() else f"p_{name}"


@dataclass
class _Symbols:
    """Declarations collected while walking a library."""

    propositions: dict[str, str] = field(default_factory=dict)  # ident -> comment
    integers: dict[str, str] = field(default_factory=dict)  # ident -> comment
    bounds: dict[str, int] = field(default_factory=dict)  # ident -> largest compared constant
    saturating: set[str] = field(default_factory=set)

    def proposition(self, ident: str, comment: str) -> str:
        self.propositions.setdefault(ident, comment)
        return ident

    def integer(self, ident: str, comment: str, *, saturating: bool) -> str:
        self.integers.setdefault(ident, comment)
        if saturating:
            self.saturating.add(ident)
        return ident

    def note_bound(self, ident: str, value: float) -> None:
        if ident in self.integers:
            self.bounds[ident] = max(self.bounds.get(ident, 0), int(value) + 1)


def _term(term: Any, sym: _Symbols) -> tuple[str, str | None]:
    """Return ``(text, variable_ident_or_None)`` for a numeric term."""
    if isinstance(term, Const):
        v = term.value
        return (str(int(v)) if float(v).is_integer() else str(v)), None
    if isinstance(term, Var):
        # Quantities written in the infix syntax arrive as ``Var`` nodes named
        # after the quantity; give them the same identifiers as the typed terms.
        if term.name in ("arg_value", "arg_numeric", "ArgValue") and len(term.args) == 2:
            ident = _ident("num", *term.args)
            return sym.integer(
                ident, f"arg_numeric({term.args[0]}, {term.args[1]})", saturating=False
            ), ident
        if term.name in ("arg_length", "ArgLength") and len(term.args) == 2:
            ident = _ident("len", *term.args)
            return sym.integer(
                ident, f"arg_length({term.args[0]}, {term.args[1]})", saturating=False
            ), ident
        if term.name in ("ctx_value", "CtxValue") and len(term.args) == 1:
            ident = _ident("ctx", term.args[0])
            return sym.integer(ident, f"ctx_value({term.args[0]})", saturating=False), ident
        ident = _ident("v", term.name, *term.args)
        return sym.integer(ident, term.key(), saturating=term.name in _SATURATING), ident
    if isinstance(term, ArgValue):
        ident = _ident("num", term.tool, term.field)
        return sym.integer(
            ident, f"arg_numeric({term.tool}, {term.field})", saturating=False
        ), ident
    if isinstance(term, ArgLength):
        ident = _ident("len", term.tool, term.field)
        return sym.integer(ident, f"arg_length({term.tool}, {term.field})", saturating=False), ident
    if isinstance(term, CtxValue):
        ident = _ident("ctx", term.key)
        return sym.integer(ident, f"ctx_value({term.key})", saturating=False), ident
    if isinstance(term, UnaryFn):
        inner, _ = _term(term.arg, sym)
        ident = _ident("fn", term.name, inner)
        return sym.integer(ident, f"{term.name}({inner})", saturating=False), ident
    raise ChaseExportError(f"unsupported numeric term {type(term).__name__}")


def _relation(node: Any, op: str, sym: _Symbols) -> str:
    left, lvar = _term(node.left, sym)
    right, rvar = _term(node.right, sym)
    if lvar and isinstance(node.right, Const):
        sym.note_bound(lvar, node.right.value)
    if rvar and isinstance(node.left, Const):
        sym.note_bound(rvar, node.left.value)
    return f"({left} {op} {right})"


def _formula(node: Any, sym: _Symbols, finite: bool) -> str:
    """Translate one formula node into CHASE logics syntax."""
    if isinstance(node, Atom):
        ident = _ident("p", node.predicate, *node.args)
        return sym.proposition(ident, node.key())
    if isinstance(node, Subset):
        ident = _ident("subset", node.left, node.right)
        return sym.proposition(ident, f"subset({node.left}, {node.right})")
    if isinstance(node, Not):
        return f"(! {_formula(node.child, sym, finite)})"
    if isinstance(node, And):
        return f"({_formula(node.left, sym, finite)} /\\ {_formula(node.right, sym, finite)})"
    if isinstance(node, Or):
        return f"({_formula(node.left, sym, finite)} \\/ {_formula(node.right, sym, finite)})"
    if isinstance(node, Implies):
        return f"({_formula(node.left, sym, finite)} -> {_formula(node.right, sym, finite)})"
    if isinstance(node, G):
        body = _formula(node.child, sym, finite)
        return f"(G (alive -> {body}))" if finite else f"(G {body})"
    if isinstance(node, F):
        body = _formula(node.child, sym, finite)
        return f"(F (alive /\\ {body}))" if finite else f"(F {body})"
    if isinstance(node, X):
        body = _formula(node.child, sym, finite)
        return f"(X (alive -> {body}))" if finite else f"(X {body})"
    if isinstance(node, U):
        left = _formula(node.left, sym, finite)
        right = _formula(node.right, sym, finite)
        return f"({left} U (alive /\\ {right}))" if finite else f"({left} U {right})"
    if isinstance(node, Le):
        return _relation(node, "<=", sym)
    if isinstance(node, Lt):
        return _relation(node, "<", sym)
    if isinstance(node, Ge):
        return _relation(node, ">=", sym)
    if isinstance(node, Gt):
        return _relation(node, ">", sym)
    if isinstance(node, Eq):
        return _relation(node, "=", sym)
    raise ChaseExportError(f"unsupported formula node {type(node).__name__}")


def _raw(constraint: Any) -> Any:
    return getattr(constraint, "formula", constraint)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


@dataclass
class ExportedContract:
    ident: str
    desc: str
    assumptions: list[str]
    guarantees: list[str]


def export_logics(
    contracts: Sequence[Any],
    *,
    name: str = "contragent_library",
    semantics: str = "finite",
) -> str:
    """Render a contract library as a CHASE logics specification.

    Args:
        contracts: :class:`~contragent.models.contract.Contract` objects (or
            anything with ``assumptions`` / ``guarantees`` lists of
            formulas or ``DetFormula`` wrappers).
        name: The ``NAME:`` of the CHASE system.
        semantics: ``"finite"`` applies the LTL\\ :sub:`f`-to-LTL
            translation with the ``alive`` proposition; ``"infinite"``
            exports the formulas unchanged.
    """
    if semantics not in ("finite", "infinite"):
        raise ValueError("semantics must be 'finite' or 'infinite'")
    finite = semantics == "finite"
    sym = _Symbols()
    if finite:
        sym.proposition("alive", "the finite trace has not ended")
    exported: list[ExportedContract] = []
    for i, c in enumerate(contracts, start=1):
        desc = getattr(c, "desc", None) or f"contract {i}"
        ident = f"c{i}"
        assert len(ident) <= MAX_CONTRACT_IDENT
        a = [_formula(_raw(x), sym, finite) for x in _as_list(getattr(c, "assumptions", []))]
        g = [_formula(_raw(x), sym, finite) for x in _as_list(getattr(c, "guarantees", []))]
        exported.append(ExportedContract(ident, desc, a, g))

    out: list[str] = []
    out.append(f"# CHASE logics specification exported by ContrAgent ({semantics}-trace semantics)")
    system_name = _IDENT_RE.sub("_", name).strip("_") or "contragent_library"
    if not system_name[0].isalpha():
        system_name = f"lib_{system_name}"
    out.append(f"NAME: {system_name};")
    out.append("")
    out.append("# --- propositions: one per grounded interaction predicate ---")
    for ident, comment in sym.propositions.items():
        out.append(f"proposition {ident};  # {_comment(comment)}")
    if sym.integers:
        out.append("")
        out.append("# --- quantities: saturating counters carry an explicit range ---")
        for ident, comment in sym.integers.items():
            if ident in sym.saturating:
                bound = sym.bounds.get(ident, 1)
                out.append(
                    f"integer (0:{bound}) variable {ident};  # {comment}, saturates at {bound}"
                )
            else:
                out.append(f"integer variable {ident};  # {_comment(comment)}")
    out.append("")
    if finite:
        out.append("# Finite-trace axiom: a non-empty prefix, then alive is false forever.")
        out.append(f"CONTRACT {FINITE_TRACE_CONTRACT}:")
        out.append("  Assumptions:")
        out.append("    true;")
        out.append("  Guarantees:")
        out.append("    (alive /\\ (alive U (G (! alive))));")
        out.append("")
    for ec in exported:
        out.append(f"# {_comment(ec.desc)}")
        out.append(f"CONTRACT {ec.ident}:")
        # CHASE's console expects both blocks; an unconditional contract assumes true.
        out.append("  Assumptions:")
        for a in ec.assumptions or ["true"]:
            out.append(f"    {a};")
        out.append("  Guarantees:")
        for g in ec.guarantees or ["true"]:
            out.append(f"    {g};")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def export_library(
    config_path: str | Path,
    out_path: str | Path | None = None,
    *,
    agent_id: str | None = None,
    semantics: str = "finite",
) -> str:
    """Export a library file to CHASE logics syntax and optionally write it."""
    from contragent.config import load_system

    system = load_system(config_path)
    contracts = list(system.contracts)
    if agent_id:
        contracts = [c for c in contracts if c.agent.id == agent_id]
    text = export_logics(contracts, name=Path(config_path).stem, semantics=semantics)
    if out_path is not None:
        Path(out_path).write_text(text)
    return text


# ----------------------------------------------------------------------
# Optional: drive the CHASE console through its Python bindings
# ----------------------------------------------------------------------


def is_available() -> bool:
    """True when CHASE's ``pychase_logicsLang`` module can be imported."""
    return importlib.util.find_spec("pychase_logicsLang") is not None


def unavailable_reason() -> str:
    return (
        "CHASE's Python bindings are not installed. Build chase-cps/core-library "
        "and chase-cps/logics_tool with pybind11 (https://chase-cps.github.io) and put "
        "the resulting pychase / pychase_logicsLang modules on PYTHONPATH."
    )


class ChaseSession:
    """A CHASE logics console loaded with an exported specification.

    Wraps ``pychase_logicsLang.LogicsSpecsBuilder`` (parser) and
    ``pychase_logicsLang.Console`` (commands such as ``verify <contract>
    <file.smv>``, ``refinement <c1> <c2> <file>``, ``synthesize <contract>
    <file>``). Requires the CHASE bindings; see :func:`unavailable_reason`.
    """

    def __init__(self, logics_path: str | Path, workdir: str | Path | None = None) -> None:
        if not is_available():
            raise ChaseUnavailable(unavailable_reason())
        # The console module returns core types (System, Contract); importing the
        # core bindings first registers them with pybind11.
        import pychase  # type: ignore[import-not-found]  # noqa: F401
        import pychase_logicsLang as _cl  # type: ignore[import-not-found]

        self.logics_path = Path(logics_path)
        self.workdir = Path(workdir) if workdir is not None else self.logics_path.parent
        self._builder = _cl.LogicsSpecsBuilder()
        self._builder.parseSpecificationFile(str(self.logics_path))
        self.system = self._builder.getSystem()
        # The console joins its output directory and file names by concatenation.
        self._console = _cl.Console(self.system, str(self.workdir) + "/")

    def run(self, command: str) -> int:
        """Run one console command; returns the console's status code.

        The semicolon that terminates commands in a ``logics_tool`` command
        file is stripped by that tool before the console sees the command;
        the console API takes the bare command, so a trailing one is removed.
        """
        return self._console.run(command.strip().rstrip(";").strip())

    def verify(
        self,
        contract: str,
        out_file: str,
        *,
        declare: bool = True,
        int_range: tuple[int, int] | None = None,
    ) -> int:
        """Emit the NuSMV model of ``contract`` (CHASE ``verify``).

        The model negates the assumption and the guarantee as two ``LTLSPEC``
        properties, so a property reported false by nuXmv (with a
        counterexample) means the formula is satisfiable. With ``declare``
        the ``VAR`` block, which CHASE leaves empty for specifications read
        from a logics file, is filled from the declarations of the export;
        see :func:`smv_declarations` for ``int_range``.
        """
        if "smv" not in out_file:  # mirrors the console's naming rule
            out_file += ".smv"
        rc = self.run(f"verify {contract} {out_file}")
        smv = self.workdir / out_file
        if declare and smv.exists():
            lines = smv_declarations(self.logics_path.read_text(), int_range=int_range)
            text = smv.read_text()
            if lines and "\nVAR\n" in text:
                smv.write_text(text.replace("\nVAR\n", "\nVAR\n" + "".join(lines), 1))
        return rc

    def refinement(self, refined: str, abstract: str, out_file: str) -> int:
        """Emit the refinement-check model for ``refined`` <= ``abstract``.

        CHASE's console crashes here on specifications read from a logics
        file (the parser leaves the declarations on the system rather than on
        the contracts); :meth:`PychaseTranslator.refines` is the working path.
        """
        return self.run(f"refinement {refined} {abstract} {out_file}")


def contract_identifiers(text: str) -> list[str]:
    """The ``CONTRACT`` identifiers declared in an exported specification."""
    return re.findall(r"^CONTRACT\s+([A-Za-z][A-Za-z0-9_]*):", text, flags=re.M)


_DECL_RE = re.compile(
    r"^(?P<kind>proposition|integer(?: \((?P<lo>\d+):(?P<hi>\d+)\))? variable) (?P<name>[A-Za-z]\w*);",
    flags=re.M,
)


def smv_declarations(text: str, *, int_range: tuple[int, int] | None = None) -> list[str]:
    """NuSMV ``VAR`` lines for the declarations of an exported specification.

    CHASE's NuSMV printer writes only the variables attached to the contract,
    and its logics parser attaches declarations to the system, so the model
    that ``verify`` emits has an empty ``VAR`` block. Propositions become
    booleans and ranged integers keep their range. An unbounded integer gets
    ``int_range`` when one is given (nuXmv's BDD algorithms need finite
    domains) and the ``integer`` type otherwise (for its SMT-based
    algorithms, e.g. ``msat_check_ltlspec_bmc``).
    """
    lines: list[str] = []
    for m in _DECL_RE.finditer(text):
        if m["kind"] == "proposition":
            typ = "boolean"
        elif m["lo"] is not None:
            typ = f"{m['lo']}..{m['hi']}"
        elif int_range is not None:
            typ = f"{int_range[0]}..{int_range[1]}"
        else:
            typ = "integer"
        lines.append(f"\t{m['name']} : {typ};\n")
    return lines


__all__ = [
    "FINITE_TRACE_CONTRACT",
    "MAX_CONTRACT_IDENT",
    "ChaseExportError",
    "ChaseUnavailable",
    "ChaseSession",
    "export_logics",
    "export_library",
    "contract_identifiers",
    "smv_declarations",
    "is_available",
    "unavailable_reason",
]


# ----------------------------------------------------------------------
# Optional: build CHASE contracts directly through ``pychase``
# ----------------------------------------------------------------------


class PychaseTranslator:
    """Translate contracts into ``pychase`` objects for CHASE's contract algebra.

    Uses the same grounding as :func:`export_logics`: each instantiated
    predicate becomes a boolean variable named by its key, and every
    comparison is abstracted to a boolean proposition named by its canonical
    text, so the propositional-temporal structure that CHASE's logic-domain
    algebra reasons over is preserved. Reuse one translator across the
    contracts you intend to compose so shared atoms map to the same variable.
    Requires ``pychase`` (chase-cps/core-library built with pybind11).
    """

    def __init__(self) -> None:
        if importlib.util.find_spec("pychase") is None:
            raise ChaseUnavailable(unavailable_reason())
        import pychase  # type: ignore[import-not-found]

        self._rep = pychase.representation
        self._u = pychase.utilities
        self._logic = self._rep.semantic_domain.logic
        self.variables: dict[str, Any] = {}
        self.names: dict[str, str] = {}

    def _prop(self, key: str, comment: str) -> Any:
        if key not in self.variables:
            name = _ident("p", comment)
            suffix = 2
            while name in self.names and self.names[name] != key:
                name = f"{_ident('p', comment)}_{suffix}"
                suffix += 1
            self.variables[key] = self._rep.Variable(self._rep.Boolean(), self._rep.Name(name))
            self.names[name] = key
        return self._u.Prop(self.variables[key])

    def formula(self, node: Any) -> Any:
        """Translate one formula node into a CHASE ``LogicFormula``."""
        u = self._u
        if isinstance(node, Atom):
            return self._prop(node.key(), f"{node.predicate}_{'_'.join(node.args)}")
        if isinstance(node, Subset):
            return self._prop(repr(node), f"subset_{node.left}_{node.right}")
        if isinstance(node, Not):
            return u.Not(self.formula(node.child))
        if isinstance(node, And):
            return u.And(self.formula(node.left), self.formula(node.right))
        if isinstance(node, Or):
            return u.Or(self.formula(node.left), self.formula(node.right))
        if isinstance(node, Implies):
            return u.Implies(self.formula(node.left), self.formula(node.right))
        if isinstance(node, G):
            return u.Always(self.formula(node.child))
        if isinstance(node, F):
            return u.Eventually(self.formula(node.child))
        if isinstance(node, X):
            return u.Next(self.formula(node.child))
        if isinstance(node, U):
            return u.Until(self.formula(node.left), self.formula(node.right))
        if isinstance(node, (Le, Lt, Ge, Gt, Eq)):
            # predicate abstraction: the comparison is an opaque proposition
            sym = _Symbols()
            text = _relation(node, {Le: "<=", Lt: "<", Ge: ">=", Gt: ">", Eq: "="}[type(node)], sym)
            return self._prop(repr(node), text)
        raise ChaseExportError(f"unsupported formula node {type(node).__name__}")

    def _conjoin(self, items: list) -> Any | None:
        formulas = [self.formula(_raw(x)) for x in items]
        if not formulas:
            return None
        return formulas[0] if len(formulas) == 1 else self._u.LargeAnd(formulas)

    def contract(self, contract: Any, name: str | None = None) -> Any:
        """A CHASE ``Contract`` with all referenced variables declared."""
        cname = name or _ident("c", getattr(contract, "desc", None) or "contract")
        c = self._rep.Contract(cname)
        g = self._conjoin(_as_list(getattr(contract, "guarantees", [])))
        if g is None:
            raise ChaseExportError(f"contract {cname!r} has no guarantee")
        c.addGuarantees(self._logic, g)
        a = self._conjoin(_as_list(getattr(contract, "assumptions", [])))
        if a is not None:
            c.addAssumptions(self._logic, a)
        for var in self.variables.values():
            c.addDeclaration(var)
        return c

    @staticmethod
    def shared_variables(a: Any, b: Any) -> dict[str, str]:
        """Identity correspondences over the variables two CHASE contracts share.

        CHASE keeps each contract's declarations separate and treats a name
        that appears in both as a clash unless the correspondence is given.
        """
        names_a = {d.getName().getString() for d in a.declarations}
        names_b = {d.getName().getString() for d in b.declarations}
        return {n: n for n in names_a & names_b}

    def _fold(self, op: str, contracts: Sequence[Any], name: str) -> Any:
        chase = [
            self.contract(c, _ident(f"c{i}", getattr(c, "desc", "") or ""))
            for i, c in enumerate(contracts, 1)
        ]
        for cc in chase:
            self._rep.Contract.saturate(cc)
        fn = getattr(self._rep.Contract, op)
        acc = chase[0]
        for cc in chase[1:]:
            acc = fn(acc, cc, self.shared_variables(acc, cc), name)
        return acc

    def conjoin(self, contracts: Sequence[Any], name: str = "library") -> Any:
        """Saturate every contract and fold CHASE ``conjunction`` over the library.

        Conjunction is the operation for several contracts on the *same*
        component, which is what a library of contracts on one agent is.
        """
        return self._fold("conjunction", contracts, name)

    def compose(self, contracts: Sequence[Any], name: str = "library") -> Any:
        """Saturate every contract and fold CHASE ``composition`` over the library."""
        return self._fold("composition", contracts, name)

    def refines(self, refined: Any, abstract: Any, *, name: str = "refinement") -> Any:
        """Build CHASE's refinement-check contract for ``refined`` <= ``abstract``.

        ``Contract.refinementCheck`` returns a contract whose assumption is
        ``A_abstract -> A_refined`` and whose guarantee is
        ``G_refined -> G_abstract`` over the variables the two share; the
        refinement holds when both are valid, which CHASE's back ends decide.
        Use this path rather than the console's ``refinement`` command,
        which crashes on specifications read from a logics file.
        """
        a = self.contract(refined, "refined")
        b = self.contract(abstract, "abstract")
        return self._rep.Contract.refinementCheck(a, b, self.shared_variables(a, b), name)


__all__ += ["PychaseTranslator"]
