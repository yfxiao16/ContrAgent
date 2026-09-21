"""Contract-library conflict checking.

Implements the library check from the ContrAgent paper (§ Contract
library). An assumption states what the environment is required to
keep, so every assumption holds on every legal run. Contracts that are
consistent and compatible one at a time can still be unsatisfiable
together, so the library is checked as a whole, once at load time and
off the hot path:

Is :math:`\\bigwedge_i (A_i \\wedge G_i)` satisfiable? A model is one
session that keeps every assumption and meets every guarantee, which
settles the library in a single query. When it is not, a **minimal
unsatisfiable core** (Roveri et al. 2024; Ielo et al. 2026,
``mus2muc``) names the contracts that clash rather than the whole
unsatisfiable set. Assumptions that cannot hold together are caught by
the same query, since they make every extension of them unsatisfiable
too.

Backends
--------
* ``"native"`` — pure-Python, zero extra dependencies. LTLf
  satisfiability comes from :func:`contragent.formulas.sat.is_satisfiable`
  and one core per iteration is extracted with the classic
  deletion-based shrink. Disjoint cores are found by removing each
  classified core from the working set and repeating; overlapping
  cores may be missed (see ``ConflictReport.exhaustive``).
* ``"mus2muc"`` — delegates core *enumeration* to ``mus2muc``
  (Ielo, Mazzotta, Peñaloza & Ricca, "Enumerating Minimal
  Unsatisfiable Cores of LTLf Formulae", AAAI 2026;
  https://github.com/ainnoot/mus2muc), which enumerates **all** MUCs.
  Requires the ``mus2muc`` package plus its patched ``wasp`` and
  ``aaltaf``/``black`` executables. Step 2 (assumption joint-SAT)
  still runs on the native engine.
* ``"auto"`` (default) — ``mus2muc`` when its toolchain is installed,
  otherwise ``native``.

Witness-trace fast path (native backend)
----------------------------------------
Before the native automata search runs, a handful of short
grounding-consistent valuation traces (the all-false trace, plus one
single-``called(t)`` trace per call atom) are evaluated against the
conjunction of every contract using the sat engine's own progression
and weak-finalization rules, so the finite-trace semantics agrees
with the search by construction. Any trace satisfying all of them is
a **model** of the full conjunction — a satisfiability certificate
proving no unsatisfiable core exists — so the library is certified
conflict-free *searchlessly*
(``ConflictReport.certificate == "witness-trace"``). This is what
saves large all-safety libraries whose comparison propositions blow
the search's alphabet budget: they are satisfied by the all-false
trace outright. When no witness works, the check falls through to the
search unchanged (``certificate == "search"`` when it decides;
unknown-budget bailouts still leave ``certificate`` ``None``).

Only deterministic contracts participate; other constraints are skipped and
counted in ``ConflictReport.skipped``. Reported conflicts are always
sound (a reported core is genuinely unsatisfiable), and the check is
additionally *complete* for the fragment ContrAgent ships: numeric
comparisons are related pointwise by an SMT/interval theory checker,
and ``count``/``count_with`` registers get exact unit-increment
semantics — natively via the saturating-counter gadget
(:class:`contragent.formulas.sat._CounterGadget`), and on the
propositional ``mus2muc`` backend via :func:`counting_axioms`.
Remaining over-approximations (conflicts that may go undetected, never
falsely reported): ``consecutive_count`` resets, counter thresholds
above the gadget cap, and cross-term arithmetic beyond the configured
theory checker.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from contragent.formulas._pred_key import pred_key
from contragent.formulas.formula import (
    And,
    Atom,
    Formula,
    G,
    Implies,
    Not,
    collect_atoms,
)
from contragent.formulas.sat import is_satisfiable
from contragent.models.contract import Contract

# Grounding sets these predicates only on a tool_call event for their
# first-argument tool, so pointwise they imply ``called(tool)``.
_TOOL_SCOPED_PREDICATES = frozenset(
    {
        "called_with",
        "arg_has",
        "arg_field_has",
        "arg_length_exceeds",
        "arg_paths_within",
    }
)


# ---------------------------------------------------------------------------
# Units: one det contract, unwrapped to raw formulas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractUnit:
    """A pure-det contract flattened to raw ``Formula`` ASTs."""

    contract: Contract
    index: int
    label: str
    assumption: Formula | None  # AND of assumption formulas; None = unconditional
    guarantee: Formula  # AND of guarantee formulas

    @property
    def combined(self) -> Formula:
        """The :math:`A_i \\wedge G_i` conjunct this unit contributes."""
        if self.assumption is None:
            return self.guarantee
        return And(self.assumption, self.guarantee)


def _raw(constraint: Any) -> Formula | None:
    """Unwrap a constraint to its ``Formula`` AST (DetFormula-aware)."""
    from contragent.formulas.formula import FormulaMixin

    if isinstance(constraint, FormulaMixin):
        return constraint
    inner = getattr(constraint, "formula", None)
    if isinstance(inner, FormulaMixin):
        return inner
    return None


def _and_all(formulas: Sequence[Formula]) -> Formula:
    out = formulas[0]
    for f in formulas[1:]:
        out = And(out, f)
    return out


def _label_of(contract: Contract, index: int) -> str:
    if contract.desc:
        return str(contract.desc)
    for g in contract.guarantees:
        desc = getattr(g, "desc", None)
        if desc:
            return str(desc)
    return f"contract[{index}]"


def _units_of(contracts: Sequence[Contract]) -> tuple[list[ContractUnit], list[str]]:
    """Split contracts into checkable units and skipped labels."""
    units: list[ContractUnit] = []
    skipped: list[str] = []
    for i, c in enumerate(contracts):
        label = _label_of(c, i)
        if not c.is_pure_det:
            skipped.append(label)
            continue
        guarantees = [f for f in (_raw(g) for g in c.guarantees) if f is not None]
        assumptions = [f for f in (_raw(a) for a in c.assumptions) if f is not None]
        if not guarantees or len(guarantees) != len(c.guarantees):
            skipped.append(label)
            continue
        units.append(
            ContractUnit(
                contract=c,
                index=i,
                label=label,
                assumption=_and_all(assumptions) if assumptions else None,
                guarantee=_and_all(guarantees),
            )
        )
    return units, skipped


# ---------------------------------------------------------------------------
# Domain constraints from grounding semantics
# ---------------------------------------------------------------------------


def derive_domain_constraints(
    formulas: Iterable[Formula],
) -> tuple[list[list[str]], list[tuple[str, str]], list[Formula]]:
    """Derive constraints that are structurally true of the grounding layer.

    * All ``called(t)`` atoms are pairwise mutually exclusive — an event
      is a call to at most one tool.
    * ``called_with(t, ...)`` / ``arg_*(t, ...)`` atoms imply
      ``called(t)`` — they are only grounded on that tool's call events.
    * Any ``called(t)`` implies ``called_any``.
    * Ban-style count comparisons are linked to their call atom: the
      grounding layer increments ``count(t)`` *before* emitting the
      valuation, so at any event where ``called(t)`` holds,
      ``count(t) >= 1``. Hence ``count(t) <= 0`` (a ban on ``t``)
      is pointwise incompatible with ``called(t)``. These
      are returned as extra axiom *formulas* to conjoin with every
      satisfiability query (third element of the result).

    Only constraints among atoms actually present are emitted; they
    tighten the boolean abstraction and can only expose more genuine
    conflicts, never fabricate one.
    """
    formulas = list(formulas)
    atoms: set[Atom] = set()
    for f in formulas:
        atoms |= collect_atoms(f)
    # Counter registers are driven by call atoms the satisfiability
    # engine interns even when no formula mentions them (the
    # saturating-counter gadget needs ``called(x)`` to exercise
    # ``count(x)``). Include those *virtual* atoms so they join the
    # mutex group and implication derivation below.
    atoms |= _virtual_driver_atoms(formulas)
    keys = {a.key() for a in atoms}

    called_keys = sorted(a.key() for a in atoms if a.predicate == "called")
    mutex_groups: list[list[str]] = []
    if len(called_keys) >= 2:
        mutex_groups.append(called_keys)

    implications: list[tuple[str, str]] = []
    called_any_key = pred_key("called_any")
    for a in atoms:
        if a.predicate in _TOOL_SCOPED_PREDICATES and a.args:
            target = pred_key("called", a.args[0])
            if target in keys:
                implications.append((a.key(), target))
        if a.predicate == "called" and called_any_key in keys:
            implications.append((a.key(), called_any_key))

    axioms = _count_ban_axioms(formulas, keys) + _monotone_axioms(formulas)
    return mutex_groups, sorted(set(implications)), axioms


# Counter Vars whose value is >= 1 at any event where the corresponding
# call atom holds (grounding increments them before emitting valuations).
_COUNT_VAR_TO_CALL_ATOM = {
    "count": "called",
    "count_with": "called_with",
    "consecutive_count": "called",
}


def _collect_comparisons(formula: Any, out: list) -> None:
    from contragent.formulas.formula import (
        And as _And,
    )
    from contragent.formulas.formula import (
        Eq,
        Ge,
        Gt,
        Le,
        Lt,
    )
    from contragent.formulas.formula import F as _F
    from contragent.formulas.formula import G as _G
    from contragent.formulas.formula import (
        Implies as _Implies,
    )
    from contragent.formulas.formula import (
        Not as _Not,
    )
    from contragent.formulas.formula import (
        Or as _Or,
    )
    from contragent.formulas.formula import U as _U
    from contragent.formulas.formula import X as _X

    if isinstance(formula, (Le, Lt, Ge, Gt, Eq)):
        out.append(formula)
        return
    if isinstance(formula, (_Not, _G, _F, _X)):
        _collect_comparisons(formula.child, out)
        return
    if isinstance(formula, (_And, _Or, _Implies, _U)):
        _collect_comparisons(formula.left, out)
        _collect_comparisons(formula.right, out)


def _count_ban_axioms(formulas: list[Formula], keys: set[str]) -> list[Formula]:
    """Axioms linking ban-threshold count comparisons to call atoms.

    For a comparison whose truth is impossible at a call event —
    ``count(t) <= c`` with ``c < 1``, or ``count(t) < c`` with
    ``c <= 1`` — emit ``G(called(t) -> !(comparison))``. The comparison
    node is rebuilt in the canonical ``Le``/``Lt`` orientation so the
    boolean abstraction maps it to the same proposition as in the
    contract (``Gt``/``Ge`` share that proposition with flipped
    polarity). The axiom is only emitted when the call atom itself
    occurs in the library, so it never grows the alphabet on its own.
    """
    from contragent.formulas.formula import Const, Ge, Gt, Le, Lt, Var

    comparisons: list = []
    for f in formulas:
        _collect_comparisons(f, comparisons)

    axioms: list[Formula] = []
    seen: set[tuple] = set()
    for cmp_node in comparisons:
        if isinstance(cmp_node, (Le, Gt)):
            op = "le"
        elif isinstance(cmp_node, (Lt, Ge)):
            op = "lt"
        else:
            continue
        left, right = cmp_node.left, cmp_node.right
        if not isinstance(left, Var) or not isinstance(right, Const):
            continue
        call_pred = _COUNT_VAR_TO_CALL_ATOM.get(left.name)
        if call_pred is None or not left.args:
            continue
        if not isinstance(right.value, (int, float)):
            continue
        impossible_at_call = (op == "le" and right.value < 1) or (op == "lt" and right.value <= 1)
        if not impossible_at_call:
            continue
        call_atom = Atom(call_pred, *left.args)
        if call_atom.key() not in keys:
            continue
        dedup = (op, left.name, left.args, right.value)
        if dedup in seen:
            continue
        seen.add(dedup)
        canonical = Le(left, right) if op == "le" else Lt(left, right)
        axioms.append(G(Implies(call_atom, Not(canonical))))
    return axioms


# Registers that never decrease within a session (grounding only ever
# increments them). ``consecutive_count`` is excluded — it resets when a
# different tool is called.
_MONOTONE_VAR_NAMES = frozenset({"count", "count_with"})


def _virtual_driver_atoms(formulas: list[Formula]) -> set[Atom]:
    """Call atoms that drive counter registers appearing in comparisons."""
    from contragent.formulas.formula import Const, Var
    from contragent.formulas.sat import _GADGET_VAR_TO_DRIVER

    comparisons: list = []
    for f in formulas:
        _collect_comparisons(f, comparisons)
    out: set[Atom] = set()
    for cmp_node in comparisons:
        left, right = cmp_node.left, cmp_node.right
        if not isinstance(left, Var) or not isinstance(right, Const):
            continue
        driver_pred = _GADGET_VAR_TO_DRIVER.get(left.name)
        if driver_pred and left.args and isinstance(right.value, (int, float)):
            out.add(Atom(driver_pred, *left.args))
    return out


def counting_axioms(formulas: Iterable[Formula], cap: int = 64) -> list[Formula]:
    """Exact LTLf characterization of counter thresholds.

    The native satisfiability engine models ``count``/``count_with``
    registers exactly via its saturating-counter gadget; purely
    propositional backends (``mus2muc``) cannot, so this emits the same
    semantics as formulas. For a threshold comparison ``p`` that means
    "at most k driver events have occurred" (``count <= c`` with
    ``k = floor(c)``; ``count < c`` with ``k = ceil(c) - 1``), the
    characterization is a chain of k+1 weak-until layers over the
    driver atom ``d``::

        level_k   = (p ∧ ¬d) W (d ∧ ¬p ∧ X G(¬p))
        level_j   = (p ∧ ¬d) W (p ∧ d ∧ X level_{j+1})    (j < k)
        axiom     = level_0

    — ``p`` holds through the first k driver events and flips to false
    at the (k+1)-th, forever. ``k < 0`` degenerates to ``G(¬p)``.
    Thresholds above ``cap`` are skipped (the formula grows linearly in
    k); equality comparisons are also skipped. Both remain sound —
    just unconstrained — for the propositional backend.
    """
    import math

    from contragent.formulas.formula import Const, Ge, Gt, Le, Lt, Or, U, Var, X
    from contragent.formulas.sat import _GADGET_VAR_TO_DRIVER

    def weak_until(a: Formula, b: Formula) -> Formula:
        return Or(U(a, b), G(a))

    comparisons: list = []
    for f in formulas:
        _collect_comparisons(f, comparisons)

    axioms: list[Formula] = []
    seen: set[tuple] = set()
    for cmp_node in comparisons:
        if isinstance(cmp_node, (Le, Gt)):
            op = "le"
        elif isinstance(cmp_node, (Lt, Ge)):
            op = "lt"
        else:
            continue  # eq: rare, handled exactly only by the native gadget
        left, right = cmp_node.left, cmp_node.right
        if not isinstance(left, Var) or not isinstance(right, Const):
            continue
        driver_pred = _GADGET_VAR_TO_DRIVER.get(left.name)
        if driver_pred is None or not left.args:
            continue
        if not isinstance(right.value, (int, float)):
            continue
        dedup = (op, left.name, left.args, right.value)
        if dedup in seen:
            continue
        seen.add(dedup)
        c = float(right.value)
        k = math.floor(c) if op == "le" else math.ceil(c) - 1
        p = Le(left, right) if op == "le" else Lt(left, right)
        if k < 0:
            axioms.append(G(Not(p)))  # counters are never negative
            continue
        if k > cap:
            continue
        d = Atom(driver_pred, *left.args)
        level: Formula = weak_until(And(p, Not(d)), And(And(d, Not(p)), X(G(Not(p)))))
        for _ in range(k):
            level = weak_until(And(p, Not(d)), And(And(p, d), X(level)))
        axioms.append(level)
    return axioms


def _monotone_axioms(formulas: list[Formula]) -> list[Formula]:
    """Persistence axioms for monotone counters (cross-step arithmetic).

    The pointwise theory checker in :mod:`contragent.formulas.theory`
    relates comparisons *within* one event; it cannot know that
    ``count(t)`` never decreases *across* events. For each upper-bound
    comparison over a monotone register, emit the sound axiom

        ``G(!(count <= c) -> G(!(count <= c)))``

    — once the counter has exceeded a bound it stays above it — so
    e.g. ``F(count >= 2 followed by count <= 1)`` shapes become
    unsatisfiable. The comparison is rebuilt in canonical ``Le``/``Lt``
    orientation to share its abstract proposition with the contracts.
    """
    from contragent.formulas.formula import Const, Ge, Gt, Le, Lt, Var

    comparisons: list = []
    for f in formulas:
        _collect_comparisons(f, comparisons)

    axioms: list[Formula] = []
    seen: set[tuple] = set()
    for cmp_node in comparisons:
        if isinstance(cmp_node, (Le, Gt)):
            op = "le"
        elif isinstance(cmp_node, (Lt, Ge)):
            op = "lt"
        else:
            continue
        left, right = cmp_node.left, cmp_node.right
        if not isinstance(left, Var) or not isinstance(right, Const):
            continue
        if left.name not in _MONOTONE_VAR_NAMES:
            continue
        dedup = (op, left.name, left.args, right.value)
        if dedup in seen:
            continue
        seen.add(dedup)
        canonical = Le(left, right) if op == "le" else Lt(left, right)
        axioms.append(G(Implies(Not(canonical), G(Not(canonical)))))
    return axioms


# ---------------------------------------------------------------------------
# Witness-trace SAT certificate (searchless fast path)
# ---------------------------------------------------------------------------

# How many single-called-atom witness traces to try before giving up and
# falling through to the automata search.
_WITNESS_CALLED_CAP = 50


def _witness_traces(
    formulas: Sequence[Formula], cap: int = _WITNESS_CALLED_CAP
) -> Iterator[list[dict[str, object]]]:
    """Yield cheap grounding-consistent valuation traces to try as witnesses.

    Candidates, all of length 1:

    * the **all-false** valuation — no tool called, every counter 0
      (a real non-tool-call event grounds exactly like this);
    * one trace per distinct ``called(t)`` atom (capped at *cap*) in
      which exactly that atom is true — mirroring the valuation the
      grounding layer emits for a first call to ``t``: ``called_any``
      set alongside it, and ``count(t)`` / ``consecutive_count(t)``
      already advanced to 1 (grounding increments counters *before*
      emitting the valuation).

    Construction keeps every candidate inside the sat engine's
    domain-constrained trace space: at most one ``called`` atom per
    step (mutex), tool-scoped predicates all false (implications hold
    vacuously), and counter registers agreeing with the
    saturating-counter gadget.
    """
    yield [{}]
    atoms: set[Atom] = set()
    for f in formulas:
        atoms |= collect_atoms(f)
    # Counter comparisons pull in their virtual driver atoms so e.g.
    # ``F(count(x) >= 1)`` gets a ``called(x)`` witness candidate even
    # when no formula mentions the call atom itself.
    atoms |= _virtual_driver_atoms(list(formulas))
    called_any_key = pred_key("called_any")
    tools = sorted({a.args[0] for a in atoms if a.predicate == "called" and a.args})
    for tool in tools[:cap]:
        yield [
            {
                pred_key("called", tool): True,
                called_any_key: True,
                pred_key("count", tool): 1,
                pred_key("consecutive_count", tool): 1,
            }
        ]


def _witness_certifies_sat(units: Sequence[ContractUnit], axioms: Sequence[Formula]) -> bool:
    """Try short witness traces against the conjunction of all units.

    A trace that satisfies every unit's combined formula is a *model*
    of :math:`\\bigwedge_i (A_i \\wedge G_i)` — and a model of the whole
    conjunction is a model of every subset, so **no unsatisfiable core
    can exist**. That certifies the library conflict-free without the
    automata search and its state/alphabet budgets, which
    comparison-heavy libraries routinely exceed. All-safety libraries
    (no eventuality obligations) are certified by the all-false trace
    immediately.

    Evaluation runs the sat engine's **own** abstraction, progression,
    and weak end-of-trace finalization
    (:func:`contragent.formulas.sat._abstract` /
    :func:`~contragent.formulas.sat._progress` /
    :func:`~contragent.formulas.sat._finalize`) on the conjunction,
    projecting each concrete witness state onto an abstract valuation
    (atoms by ``pred_key`` lookup; comparison propositions by concrete
    evaluation with the runtime's missing → ``False`` resolution). So
    the semantics — ``G`` vacuous at trace end, ``F`` fails if never
    satisfied, ``X`` weak — agrees with the search **by construction**.
    That exactness matters: the two general-purpose evaluators both
    diverge from the search on end-of-trace edge cases and could
    certify SAT where the search proves a genuine conflict — the
    stateless recursive evaluator's weak ``X`` is trivially true *at*
    the last position without imposing its child on the empty suffix
    (``X(F(p))`` "holds" on a one-event trace), and ``DFAEvaluator``
    finalizes a residual ``Not(atom)`` to ⊤ where the search's
    NNF'd negative literal collapses to ⊥. A failed witness only ever
    falls through to the search, never fabricates a verdict.

    The projected valuations stay inside the search's constrained
    space: comparisons over partial terms (``ArgValue``, ``CtxValue``,
    …) resolve missing → ``False``, matching the theory layer's rule
    that negative literals over partial terms carry no constraint;
    ``Var`` terms resolve to concrete numbers, so the induced
    comparison assignment is pointwise theory-consistent and agrees
    with the saturating-counter gadget (the witness carries the true
    register values). The domain ``axioms`` are conjoined alongside
    the units exactly as in the search's queries.
    """
    from contragent.formulas import sat as _sat
    from contragent.formulas.dfa_evaluator import _resolve_arith, _safe_compare

    formulas = [u.combined for u in units] + list(axioms)
    table = _sat._LeafTable()
    root = _sat._mk_and(_sat._abstract(f, table) for f in formulas)
    if isinstance(root, bool):
        return root

    def val_of(state: dict[str, object]) -> frozenset[int]:
        true_ids: set[int] = set()
        for key, pid in table.atom_ids.items():
            if state.get(key, False):
                true_ids.add(pid)
        for pid, (op, left, right, _total) in table.cmp_info.items():
            if _safe_compare(op, _resolve_arith(left, state), _resolve_arith(right, state)):
                true_ids.add(pid)
        return frozenset(true_ids)

    for trace in _witness_traces(formulas):
        node: Any = root
        for state in trace:
            node = _sat._progress(node, val_of(state))
            if node is False:
                break
        else:
            if _sat._finalize(node) is True:
                return True
    return False


# ---------------------------------------------------------------------------
# Deletion-based MUC extraction
# ---------------------------------------------------------------------------


def extract_muc(
    units: Sequence[ContractUnit],
    sat_of: Callable[[Sequence[ContractUnit]], bool | None],
) -> tuple[list[ContractUnit], bool]:
    """Shrink an unsatisfiable set of units to a minimal core.

    Classic deletion-based algorithm: drop one element at a time, keep
    the drop whenever the remainder stays unsatisfiable. Precondition:
    ``sat_of(units) is False``.

    Returns ``(core, minimal)`` where ``minimal`` is False when some
    oracle call returned unknown — the core is still unsatisfiable but
    may not be minimal.
    """
    core = list(units)
    minimal = True
    i = 0
    while i < len(core):
        candidate = core[:i] + core[i + 1 :]
        if not candidate:
            break
        verdict = sat_of(candidate)
        if verdict is False:
            core = candidate
        else:
            if verdict is None:
                minimal = False
            i += 1
    return core, minimal


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConflictCore:
    """One minimal unsatisfiable core of :math:`\\bigwedge (A_i \\wedge G_i)`.

    Attributes:
        units: The contracts in the core.
        minimal: False when core shrinking hit an unknown oracle call
            (the core is unsatisfiable but possibly non-minimal).
    """

    units: tuple[ContractUnit, ...]
    minimal: bool = True

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(u.label for u in self.units)


@dataclass
class ConflictReport:
    """Outcome of a library conflict check.

    ``ok`` is True when no genuine conflict was found and no
    satisfiability query came back unknown.

    ``certificate`` records how the native backend settled the verdict:
    ``"witness-trace"`` — a concrete short trace satisfying every
    contract certified the library conflict-free without entering the
    automata search; ``"search"`` — the automata search decided (either
    way); ``None`` — no native certificate (trivial <2-unit library,
    mus2muc backend, or an unknown-budget bailout).
    """

    conflicts: list[ConflictCore] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    checked: int = 0
    unknown: bool = False
    exhaustive: bool = False
    backend: str = "native"
    certificate: str | None = None

    @property
    def ok(self) -> bool:
        return not self.conflicts and not self.unknown

    def render(self) -> str:
        """Human-readable multi-line summary."""
        lines: list[str] = []
        if self.ok:
            cert = ", witness-trace certificate" if self.certificate == "witness-trace" else ""
            lines.append(
                f"conflict-free: {self.checked} det contract(s) checked "
                f"({self.backend} backend{cert})"
            )
        else:
            lines.append(
                f"{len(self.conflicts)} conflict(s) among {self.checked} det "
                f"contract(s) ({self.backend} backend)"
            )
        for n, core in enumerate(self.conflicts, 1):
            note = "" if core.minimal else " [core may not be minimal]"
            lines.append(f"  conflict {n}{note}:")
            for label in core.labels:
                lines.append(f"    - {label}")
        if self.skipped:
            lines.append(
                f"  skipped {len(self.skipped)} non-det contract(s): " + ", ".join(self.skipped)
            )
        if self.unknown:
            lines.append(
                "  note: a satisfiability query exceeded its search budget; "
                "the library may contain undetected conflicts"
            )
        if not self.exhaustive and self.conflicts:
            lines.append(
                "  note: native backend reports disjoint cores; install "
                "mus2muc (github.com/ainnoot/mus2muc) to enumerate all cores"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def check_conflicts(
    contracts: Sequence[Contract],
    *,
    backend: str = "auto",
    max_states: int = 50_000,
    max_alphabet: int = 4096,
    theory_checker: Any = "auto",
    exact_counters: bool = True,
    mus2muc_bin_folder: Any = None,
    mus2muc_timeout: int = 60,
) -> ConflictReport:
    """Check that a contract library is conflict-free.

    Args:
        contracts: The loaded library (e.g. ``guard._system._contracts``
            or any list of :class:`~contragent.models.contract.Contract`).
        backend: ``"auto"``, ``"native"``, or ``"mus2muc"`` (see module
            docstring). ``"mus2muc"`` raises if the Ielo et al.
            toolchain is not installed.
        max_states / max_alphabet: Budgets for the native satisfiability
            search; exceeding either marks the report ``unknown``.
        theory_checker: Pointwise arithmetic reasoning for numeric
            contracts (see :func:`contragent.formulas.sat.is_satisfiable`).
            ``"auto"`` uses z3 when installed, else the built-in
            interval checker; ``None`` disables theory reasoning.
        exact_counters: Model ``count``/``count_with`` registers exactly
            (native backend: saturating-counter gadget; mus2muc backend:
            :func:`counting_axioms`). Disable to fall back to the pure
            abstraction.
        mus2muc_bin_folder: Folder holding the ``wasp`` and
            ``aaltaf``/``black`` executables (mus2muc backend only;
            defaults to ``$CONTRAGENT_MUS2MUC_BIN`` or ``/usr/bin``).
        mus2muc_timeout: Per-enumeration timeout in seconds (mus2muc
            backend only).

    Returns:
        A :class:`ConflictReport`; the contracts that cannot hold
        together are in ``report.conflicts``.

    On the native backend the check first tries the witness-trace fast
    path (see module docstring): a cheap concrete trace satisfying
    every contract certifies the library conflict-free without the
    automata search, setting ``report.certificate = "witness-trace"``.
    Otherwise the search runs as before and sets ``certificate =
    "search"`` when it reaches a verdict (``None`` on an
    unknown-budget bailout).
    """
    if backend not in ("auto", "native", "mus2muc"):
        raise ValueError(f"backend must be auto|native|mus2muc, got {backend!r}")

    units, skipped = _units_of(contracts)
    report = ConflictReport(skipped=skipped, checked=len(units))

    if len(units) < 2:
        report.exhaustive = True
        return report

    if theory_checker == "auto":
        from contragent.formulas.theory import default_theory_checker

        theory_checker = default_theory_checker()

    mutex_groups, implications, axioms = derive_domain_constraints([u.combined for u in units])

    def sat_of(subset: Sequence[ContractUnit]) -> bool | None:
        # Domain axioms hold on every real trace, so conjoining them
        # into each query keeps UNSAT verdicts sound and cores minimal
        # *modulo the grounding semantics*.
        return is_satisfiable(
            [u.combined for u in subset] + axioms,
            mutex_groups=mutex_groups,
            implications=implications,
            max_states=max_states,
            max_alphabet=max_alphabet,
            theory_checker=theory_checker,
            exact_counters=exact_counters,
        )

    if backend in ("auto", "mus2muc"):
        from contragent.analysis import mus2muc_backend

        if mus2muc_backend.is_available(bin_folder=mus2muc_bin_folder):
            # mus2muc is purely propositional: the native gadget's exact
            # counter semantics is shipped to it as LTLf counting axioms.
            counter_axioms = counting_axioms([u.combined for u in units]) if exact_counters else []
            cores = mus2muc_backend.enumerate_mucs(
                units,
                mutex_groups=mutex_groups,
                implications=implications,
                extra_formulas=axioms + counter_axioms,
                theory_checker=theory_checker,
                bin_folder=mus2muc_bin_folder,
                timeout=mus2muc_timeout,
            )
            report.backend = "mus2muc"
            report.exhaustive = True
            for core_units in cores:
                report.conflicts.append(ConflictCore(units=tuple(core_units)))
            return report
        if backend == "mus2muc":
            raise mus2muc_backend.Mus2mucUnavailable(
                mus2muc_backend.unavailable_reason(bin_folder=mus2muc_bin_folder)
            )

    # --- Native backend ---
    # Searchless fast path: a short trace satisfying every combined
    # formula is a SAT certificate for the whole conjunction, so no
    # unsatisfiable core exists and the automata search (whose alphabet
    # budget comparison-heavy libraries exceed instantly) never runs.
    # See _witness_certifies_sat for the soundness argument.
    if _witness_certifies_sat(units, axioms):
        report.certificate = "witness-trace"
        # A model of the conjunction rules out every core, so core
        # enumeration is trivially exhaustive.
        report.exhaustive = True
        return report

    working = list(units)
    while len(working) >= 2:
        verdict = sat_of(working)
        if verdict is True:
            break
        if verdict is None:
            report.unknown = True
            break
        core_units, minimal = extract_muc(working, sat_of)
        report.conflicts.append(ConflictCore(units=tuple(core_units), minimal=minimal))
        # Remove the classified core and keep scanning for disjoint
        # cores. Overlapping cores are missed by design (see module
        # docstring); ``mus2muc`` enumerates them all.
        in_core = {id(u) for u in core_units}
        working = [u for u in working if id(u) not in in_core]
    if not report.unknown:
        report.certificate = "search"
    return report
