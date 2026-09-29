"""Finite-trace (LTLf) satisfiability for ContrAgent formulas.

This module answers one question: *does any non-empty finite trace
satisfy a conjunction of contract formulas?* It is the decision
procedure behind the library conflict check
(:mod:`contragent.analysis.conflicts`), which proceeds in two steps:
test whether :math:`\\bigwedge_i A_i` is satisfiable, so that a
legal environment exists, then extract a minimal unsatisfiable core of
:math:`\\bigwedge_i (A_i \\wedge G_i)` to name the contracts that clash.

Semantics
---------
Satisfiability is decided against the *runtime's* finite-trace
semantics — the same progression rules and weak end-of-trace collapse
implemented by :class:`contragent.formulas.dfa_evaluator.DFAEvaluator`
(``G`` → ⊤, ``F`` → ⊥, ``X`` → ⊤ weak-next, ``U`` → ⊥ at trace end).
A formula is satisfiable iff some trace of length ≥ 1 makes the
runtime monitor finish the session without a violation. Matching the
monitor, not textbook LTLf, is deliberate: a "conflict" only matters
if the enforced semantics can never be jointly met.

Abstraction
-----------
Formulas are checked over a *boolean abstraction* of their leaves:

* ``Atom`` / ``Subset`` leaves become propositions keyed by
  ``pred_key()``.
* Arithmetic comparisons become propositions. Over *total* terms
  (``Var``/``Const``) complement pairs map to one literal (``Le(l, r)``
  ↔ ``Gt(l, r)``, ``Lt(l, r)`` ↔ ``Ge(l, r)``); over *partial* terms
  (``ArgValue`` & friends, which may be missing at an event) each
  orientation keeps its own proposition, because the runtime evaluates
  both a comparison and its complement to False on a missing value.
* A pointwise **theory checker** (lazy-SMT style; see
  :mod:`contragent.formulas.theory`) prunes valuations whose comparison
  assignment is arithmetically impossible at a single event — e.g.
  ``count <= 3`` and ``count >= 10`` both true. z3 is used when
  installed, else a built-in interval checker; pass
  ``theory_checker=None`` for the pure boolean abstraction.

Cross-step counter arithmetic is handled **exactly** by the
saturating-counter gadget (:class:`_CounterGadget`): ``count``/
``count_with`` registers increment by one on their driving call atom
and are only compared against finitely many constants, so tracking
them saturated at ``max(threshold) + 1`` is an exact finitization —
no SMT needed for this fragment. The remaining over-approximations
are ``consecutive_count`` (its resets are unobservable for unmentioned
tools), registers whose thresholds exceed the gadget cap, and any
theory the pointwise checker cannot see. Callers may pass domain
constraints (``mutex_groups``, ``implications``) that are guaranteed
true of the grounding layer to tighten it further. Consequences:

* ``is_satisfiable(...) is False`` is **sound**: no real trace can
  satisfy the conjunction — a reported conflict is genuine.
* ``is_satisfiable(...) is True`` may be a spurious model — a
  theory-level conflict can be missed.
* ``None`` means the search exceeded its budget (state or alphabet
  cap) and the answer is unknown.

The search itself is an explicit reachability walk over progression
residuals: states are canonicalized residual formulas (n-ary And/Or
over ``frozenset`` so conjunct order and duplicates never mint new
states), transitions are one progression step per abstract valuation,
and a state is accepting when its weak finalization collapses to ⊤.
This is exact for the abstraction; the caps only bound runtime.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from itertools import combinations, product
from typing import Any

from contragent.formulas.formula import (
    And,
    Atom,
    Eq,
    F,
    Formula,
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
    X,
)

# ---------------------------------------------------------------------------
# Internal node algebra
# ---------------------------------------------------------------------------
#
# Residual states are canonical hashable tuples:
#
#   True / False              — decided
#   ("lit", i)                — abstract literal; i != 0, negative = negated
#   ("and", frozenset[Node])  — n-ary conjunction (flattened, deduped)
#   ("or",  frozenset[Node])  — n-ary disjunction
#   ("not", Node)             — negation of a temporal-containing node only
#                               (propositional negation is pushed to leaves)
#   ("G", Node) ("F", Node) ("X", Node) ("U", Node, Node)

Node = bool | tuple

_TEMPORAL_TAGS = ("G", "F", "X", "U")


def _mk_not(node: Node) -> Node:
    """Structural negation with De Morgan push over and/or."""
    if node is True:
        return False
    if node is False:
        return True
    tag = node[0]
    if tag == "lit":
        return ("lit", -node[1])
    if tag == "not":
        return node[1]
    if tag == "and":
        return _mk_or(_mk_not(x) for x in node[1])
    if tag == "or":
        return _mk_and(_mk_not(x) for x in node[1])
    # Temporal node: keep the negation wrapped so progression and
    # finalization compose exactly like the runtime evaluator's
    # ``Not(...)`` handling (weak-collapse duality included).
    return ("not", node)


def _mk_and(items: Iterable[Node]) -> Node:
    """Canonical n-ary conjunction: flatten, dedupe, fold constants."""
    out: set = set()
    stack = list(items)
    while stack:
        x = stack.pop()
        if x is False:
            return False
        if x is True:
            continue
        if x[0] == "and":
            stack.extend(x[1])
            continue
        out.add(x)
    for x in out:
        if _mk_not(x) in out:
            return False
    if not out:
        return True
    if len(out) == 1:
        return next(iter(out))
    return ("and", frozenset(out))


def _mk_or(items: Iterable[Node]) -> Node:
    """Canonical n-ary disjunction: flatten, dedupe, fold constants."""
    out: set = set()
    stack = list(items)
    while stack:
        x = stack.pop()
        if x is True:
            return True
        if x is False:
            continue
        if x[0] == "or":
            stack.extend(x[1])
            continue
        out.add(x)
    for x in out:
        if _mk_not(x) in out:
            return True
    if not out:
        return False
    if len(out) == 1:
        return next(iter(out))
    return ("or", frozenset(out))


# ---------------------------------------------------------------------------
# Leaf abstraction
# ---------------------------------------------------------------------------


def _is_total_term(term: Any) -> bool:
    """True when the operand always produces a value at every event.

    ``Const`` is literal; ``Var`` registers default to 0 when unset.
    Every other Term (``ArgValue``, ``CtxValue``, ``UnaryFn``, ...) may
    resolve to ``None``, in which case the runtime evaluates *both* a
    comparison and its syntactic complement to False.
    """
    from contragent.formulas.formula import Const, Var

    return isinstance(term, (Var, Const))


class _LeafTable:
    """Assigns abstract proposition ids to formula leaves.

    Comparisons over *total* terms (``Var``/``Const``) canonicalize
    complementary pairs into one proposition with opposite polarity, so
    ``Le(x, c) ∧ Gt(x, c)`` is propositionally unsatisfiable. Comparisons
    involving *partial* terms (``ArgValue`` & friends) get their own
    proposition per orientation: the runtime evaluates both ``Le(a, c)``
    and ``Gt(a, c)`` to False when the value is missing, so treating
    them as complements would exclude real traces and could fabricate
    conflicts. Their mutual exclusion (both can't be *true*) is restored
    soundly by the pointwise theory checker.

    ``cmp_info`` records ``pid -> (op, left, right, total)`` for every
    comparison proposition so the search can hand each candidate
    valuation's comparison projection to a
    :class:`~contragent.formulas.theory.TheoryChecker`.
    """

    def __init__(self) -> None:
        self._ids: dict[Any, int] = {}
        self._desc: dict[int, str] = {}
        # pred_key(...) -> id, for resolving domain constraints
        self.atom_ids: dict[str, int] = {}
        # pid -> (op, left, right, total) for comparison propositions
        self.cmp_info: dict[int, tuple[str, Any, Any, bool]] = {}

    def _intern(self, key: Any, desc: str) -> int:
        pid = self._ids.get(key)
        if pid is None:
            pid = len(self._ids) + 1
            self._ids[key] = pid
            self._desc[pid] = desc
        return pid

    _CANONICAL = {
        Le: ("le", False),
        Gt: ("le", True),
        Lt: ("lt", False),
        Ge: ("lt", True),
    }
    _ORIENTED = {Le: "le", Lt: "lt", Ge: "ge", Gt: "gt", Eq: "eq"}

    def lit_for(self, leaf: Formula, negated: bool) -> Node:
        if isinstance(leaf, (Atom, Subset)):
            key = leaf.key()
            pid = self._intern(("atom", key), key)
            self.atom_ids[key] = pid
            return ("lit", -pid if negated else pid)

        if not isinstance(leaf, (Le, Lt, Ge, Gt, Eq)):  # pragma: no cover
            raise TypeError(f"not a leaf: {type(leaf).__name__}")

        total = _is_total_term(leaf.left) and _is_total_term(leaf.right)
        if total and not isinstance(leaf, Eq):
            op, flip = self._CANONICAL[type(leaf)]
            if flip:
                negated = not negated
            pid = self._intern((op, leaf.left, leaf.right), repr(leaf))
            self.cmp_info[pid] = (op, leaf.left, leaf.right, True)
        else:
            op = self._ORIENTED[type(leaf)]
            pid = self._intern(
                (op, leaf.left, leaf.right, "total" if total else "partial"),
                repr(leaf),
            )
            self.cmp_info[pid] = (op, leaf.left, leaf.right, total)
        return ("lit", -pid if negated else pid)

    @property
    def prop_ids(self) -> list[int]:
        return sorted(self._desc)

    def describe(self, pid: int) -> str:
        return self._desc.get(abs(pid), f"p{abs(pid)}")

    def theory_literals(self, val: frozenset[int], exclude: frozenset[int] = frozenset()) -> tuple:
        """The comparison projection of a valuation, as TheoryLiterals.

        Propositions in ``exclude`` (e.g. gadget-determined counter
        comparisons) are omitted — their truth is not free, so they
        carry no information for the theory filter.
        """
        from contragent.formulas.theory import TheoryLiteral

        return tuple(
            TheoryLiteral(op=op, left=left, right=right, total=total, positive=pid in val)
            for pid, (op, left, right, total) in sorted(self.cmp_info.items())
            if pid not in exclude
        )


# ---------------------------------------------------------------------------
# Saturating-counter gadget — exact semantics for unit-increment counters
# ---------------------------------------------------------------------------

# Registers the gadget tracks exactly: incremented by exactly one on
# each event where their driver atom holds, never decremented.
# ``consecutive_count`` is excluded (it resets on other tools' calls,
# which the abstraction cannot observe for unmentioned tools).
_GADGET_VAR_TO_DRIVER = {"count": "called", "count_with": "called_with"}

# Saturation-bound cap. A comparison against a huge constant (e.g.
# ``Var('count', x) <= 10**6``) would need that many gadget states; above
# the cap the register falls back to the sound-only treatment
# (free proposition + pointwise theory + ban/persistence axioms).
_GADGET_CAP = 64


class _CounterGadget:
    """Exact finite-state semantics for counter registers.

    ``count(t)`` increments by exactly one on every ``called(t)`` event
    and is only ever compared against the finitely many constants that
    appear in the library, so tracking its value saturated at
    ``B = max(threshold) + 1`` is *exactly* equivalent to the real
    unbounded counter: all values ``>= B`` decide every comparison the
    same way, and monotonicity means the abstraction never needs to
    come back down. This turns the counter fragment from
    sound-but-incomplete (free propositions + pointwise theory) into
    sound-and-complete.

    Per register the gadget stores the driver-atom proposition id, the
    saturation bound, and ``(pid, kind, k)`` truth specs where ``kind``
    is ``"le"`` (true iff value <= k), ``"eq"`` (true iff value == k),
    or ``"false"`` (constantly false, e.g. ``count < 0`` or a
    fractional equality). During the search the state is
    ``(residual, register-values)``; each transition first advances the
    registers on the valuation's driver atoms (grounding increments
    counters *before* emitting the valuation), then injects the
    determined comparison propositions into the valuation.
    """

    def __init__(self, registers: list[tuple[int, int, list[tuple[int, str, int]]]]):
        # Each entry: (driver_pid, bound, specs)
        self._registers = registers
        self.pids: frozenset[int] = frozenset(
            pid for _, _, specs in registers for pid, _, _ in specs
        )
        self.initial: tuple[int, ...] = (0,) * len(registers)

    @classmethod
    def build(cls, table: _LeafTable, cap: int = _GADGET_CAP) -> _CounterGadget:
        import math

        from contragent.formulas.formula import Const, Var

        by_register: dict[Any, list[tuple[int, str, int]]] = {}
        for pid, (op, left, right, total) in sorted(table.cmp_info.items()):
            if not total or not isinstance(left, Var) or not isinstance(right, Const):
                continue
            if left.name not in _GADGET_VAR_TO_DRIVER or not left.args:
                continue
            if not isinstance(right.value, (int, float)):
                continue
            c = float(right.value)
            if op == "le":  # true iff value <= floor(c)
                k = math.floor(c)
                spec = (pid, "le", k) if k >= 0 else (pid, "false", 0)
            elif op == "lt":  # true iff value <= ceil(c) - 1
                k = math.ceil(c) - 1
                spec = (pid, "le", k) if k >= 0 else (pid, "false", 0)
            elif op == "eq":
                if c < 0 or c != math.floor(c):
                    spec = (pid, "false", 0)
                else:
                    spec = (pid, "eq", int(c))
            else:  # pragma: no cover - partial ops are never total
                continue
            by_register.setdefault(left, []).append(spec)

        registers: list[tuple[int, int, list[tuple[int, str, int]]]] = []
        for var, specs in by_register.items():
            bound = 1 + max((k for _, kind, k in specs if kind != "false"), default=-1)
            bound = max(bound, 1)
            if bound > cap:
                continue  # fall back to the sound-only treatment
            driver = Atom(_GADGET_VAR_TO_DRIVER[var.name], *var.args)
            driver_node = table.lit_for(driver, negated=False)
            registers.append((driver_node[1], bound, specs))
        return cls(registers)

    def step(self, regs: tuple[int, ...], val: frozenset[int]) -> tuple[int, ...]:
        """Advance the registers over one event's valuation."""
        return tuple(
            min(v + 1, bound) if driver in val else v
            for v, (driver, bound, _) in zip(regs, self._registers, strict=True)
        )

    def true_props(self, regs: tuple[int, ...]) -> frozenset[int]:
        """Comparison propositions that hold at the given register values.

        A saturated value (``v == bound``) means "actual >= bound"; every
        tracked threshold is < bound, so all specs decide to False there.
        """
        out: set[int] = set()
        for v, (_, _, specs) in zip(regs, self._registers, strict=True):
            for pid, kind, k in specs:
                if (kind == "le" and v <= k) or (kind == "eq" and v == k):
                    out.add(pid)
        return frozenset(out)


def _abstract(formula: Any, table: _LeafTable, negated: bool = False) -> Node:
    """Translate a ``Formula`` AST into the canonical node algebra (NNF)."""
    # DetFormula and friends carry the raw AST on ``.formula``.
    inner = getattr(formula, "formula", None)
    if inner is not None and not isinstance(formula, tuple):
        formula = inner

    if isinstance(formula, bool):
        return formula != negated

    if isinstance(formula, (Atom, Subset, Le, Lt, Ge, Gt, Eq)):
        return table.lit_for(formula, negated)

    if isinstance(formula, Not):
        return _abstract(formula.child, table, not negated)

    if isinstance(formula, And):
        parts = (
            _abstract(formula.left, table, negated),
            _abstract(formula.right, table, negated),
        )
        return _mk_or(parts) if negated else _mk_and(parts)

    if isinstance(formula, Or):
        parts = (
            _abstract(formula.left, table, negated),
            _abstract(formula.right, table, negated),
        )
        return _mk_and(parts) if negated else _mk_or(parts)

    if isinstance(formula, Implies):
        if negated:  # ¬(l → r) ≡ l ∧ ¬r
            return _mk_and(
                (
                    _abstract(formula.left, table, False),
                    _abstract(formula.right, table, True),
                )
            )
        return _mk_or(
            (
                _abstract(formula.left, table, True),
                _abstract(formula.right, table, False),
            )
        )

    if isinstance(formula, G):
        node: Node = ("G", _abstract(formula.child, table, False))
    elif isinstance(formula, F):
        node = ("F", _abstract(formula.child, table, False))
    elif isinstance(formula, X):
        node = ("X", _abstract(formula.child, table, False))
    elif isinstance(formula, U):
        node = (
            "U",
            _abstract(formula.left, table, False),
            _abstract(formula.right, table, False),
        )
    else:
        raise TypeError(f"is_satisfiable: unsupported formula node {type(formula).__name__}")
    return _mk_not(node) if negated else node


# ---------------------------------------------------------------------------
# Progression + weak finalization (mirrors dfa_evaluator semantics)
# ---------------------------------------------------------------------------


def _progress(node: Node, val: frozenset[int]) -> Node:
    """One progression step against a valuation (set of true prop ids)."""
    if node is True or node is False:
        return node
    tag = node[0]
    if tag == "lit":
        i = node[1]
        return (abs(i) in val) == (i > 0)
    if tag == "and":
        return _mk_and(_progress(x, val) for x in node[1])
    if tag == "or":
        return _mk_or(_progress(x, val) for x in node[1])
    if tag == "not":
        return _mk_not(_progress(node[1], val))
    if tag == "G":
        return _mk_and((_progress(node[1], val), node))
    if tag == "F":
        return _mk_or((_progress(node[1], val), node))
    if tag == "X":
        return node[1]
    if tag == "U":
        return _mk_or(
            (
                _progress(node[2], val),
                _mk_and((_progress(node[1], val), node)),
            )
        )
    raise TypeError(f"_progress: unknown node tag {tag!r}")  # pragma: no cover


def _finalize(node: Node) -> Node:
    """Collapse pending obligations at trace end (weak semantics)."""
    if node is True or node is False:
        return node
    tag = node[0]
    if tag == "lit":
        # Mirrors the runtime: a bare state-formula residual at trace
        # end cannot be evaluated without a valuation → conservative ⊥.
        return False
    if tag == "and":
        return _mk_and(_finalize(x) for x in node[1])
    if tag == "or":
        return _mk_or(_finalize(x) for x in node[1])
    if tag == "not":
        return _mk_not(_finalize(node[1]))
    if tag == "G":
        return True
    if tag == "F":
        return False
    if tag == "X":
        return True
    if tag == "U":
        return False
    raise TypeError(f"_finalize: unknown node tag {tag!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Valuation (alphabet) generation under domain constraints
# ---------------------------------------------------------------------------


def _build_alphabet(
    table: _LeafTable,
    mutex_groups: Iterable[Iterable[str]],
    implications: Iterable[tuple[str, str]],
    max_alphabet: int,
    theory: Any = None,
    exclude: frozenset[int] = frozenset(),
) -> list[frozenset[int]] | None:
    """Enumerate abstract valuations consistent with the domain constraints.

    When a ``theory`` checker is given, valuations whose comparison
    projection is pointwise theory-inconsistent (e.g. ``count <= 3`` and
    ``count >= 10`` both true) are dropped — no real trace can ground
    them, so removal is sound. Projections are cached: the checker runs
    once per distinct comparison assignment, not once per valuation.

    Returns ``None`` when the (constraint-aware) valuation count exceeds
    ``max_alphabet`` — the caller reports "unknown" rather than guessing.
    """
    all_ids = table.prop_ids

    # Resolve constraint atom keys to prop ids; drop absent atoms.
    groups: list[list[int]] = []
    assigned: set[int] = set()
    for group in mutex_groups:
        ids = [
            table.atom_ids[k]
            for k in group
            if k in table.atom_ids and table.atom_ids[k] not in assigned
        ]
        if len(ids) >= 2:
            groups.append(ids)
            assigned.update(ids)
    impl = [
        (table.atom_ids[a], table.atom_ids[b])
        for a, b in implications
        if a in table.atom_ids and b in table.atom_ids
    ]

    free = [i for i in all_ids if i not in assigned and i not in exclude]

    est = 1
    for g in groups:
        est *= len(g) + 1
    est <<= len(free)
    if est > max_alphabet:
        return None

    # Choices per mutex group: nobody, or exactly one member.
    group_choices: list[list[tuple[int, ...]]] = [[()] + [(i,) for i in g] for g in groups]
    free_choices: list[tuple[int, ...]] = [
        c for r in range(len(free) + 1) for c in combinations(free, r)
    ]

    cmp_ids = sorted(i for i in table.cmp_info if i not in exclude)
    theory_cache: dict[frozenset[int], bool] = {}

    def theory_ok(val: frozenset[int]) -> bool:
        if theory is None or not cmp_ids:
            return True
        proj = frozenset(i for i in cmp_ids if i in val)
        hit = theory_cache.get(proj)
        if hit is None:
            try:
                hit = theory.consistent(table.theory_literals(val, exclude))
            except Exception:
                hit = True  # a broken checker must never flip SAT→UNSAT
            theory_cache[proj] = hit
        return hit

    alphabet: list[frozenset[int]] = []
    for parts in product(*group_choices, free_choices):
        val = frozenset(i for part in parts for i in part)
        if any(a in val and b not in val for a, b in impl):
            continue
        if not theory_ok(val):
            continue
        alphabet.append(val)
    return alphabet


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_satisfiable(
    formulas: Any,
    *,
    mutex_groups: Iterable[Iterable[str]] = (),
    implications: Iterable[tuple[str, str]] = (),
    max_states: int = 50_000,
    max_alphabet: int = 4096,
    theory_checker: Any = "auto",
    exact_counters: bool = True,
) -> bool | None:
    """Decide whether some non-empty finite trace satisfies all ``formulas``.

    Args:
        formulas: A single ``Formula`` (or ``DetFormula``) or an iterable
            of them; iterables are conjoined.
        mutex_groups: Groups of atom ``pred_key()`` strings of which at
            most one can be true per event (e.g. all ``called(...)``
            atoms — one tool call per event).
        implications: ``(antecedent_key, consequent_key)`` pairs that
            hold pointwise in the grounding layer (e.g.
            ``called_with(t, p)`` implies ``called(t)``).
        max_states: Cap on distinct search states (progression residual
            plus counter-gadget values) to explore.
        max_alphabet: Cap on abstract valuations per step.
        theory_checker: Pointwise arithmetic reasoning over comparison
            propositions (lazy-SMT style). ``"auto"`` (default) uses z3
            when installed (``pip install contragent[smt]``), else the
            built-in interval checker; pass a
            :class:`~contragent.formulas.theory.TheoryChecker` to
            override, or ``None`` to disable and fall back to the pure
            boolean abstraction.
        exact_counters: Track ``count``/``count_with`` registers with
            the saturating-counter gadget (exact unit-increment
            semantics; see :class:`_CounterGadget`). Disable to fall
            back to treating counter comparisons as free propositions.

    Returns:
        ``True`` if a satisfying trace exists **in the abstraction**
        (exact for counter registers under the gadget cap and for the
        pointwise theory; still over-approximate for e.g.
        ``consecutive_count`` resets), ``False`` if no trace can
        satisfy the conjunction (sound — a genuine conflict), or
        ``None`` if a budget was exceeded and the answer is unknown.
    """
    if isinstance(formulas, (list, tuple, set, frozenset)):
        items = list(formulas)
    else:
        items = [formulas]

    if theory_checker == "auto":
        from contragent.formulas.theory import default_theory_checker

        theory_checker = default_theory_checker()

    table = _LeafTable()
    root = _mk_and(_abstract(f, table) for f in items)

    if root is True:
        return True
    if root is False:
        return False

    # Build the gadget before the alphabet: it interns the driver atoms
    # (e.g. ``called(x)`` for ``count(x)``) into the table so the search
    # can actually exercise the counters.
    gadget = _CounterGadget.build(table) if exact_counters else _CounterGadget([])

    alphabet = _build_alphabet(
        table,
        mutex_groups,
        implications,
        max_alphabet,
        theory=theory_checker,
        exclude=gadget.pids,
    )
    if alphabet is None:
        return None

    track_counters = bool(gadget.initial)
    init = (root, gadget.initial)
    seen: set[tuple[Node, tuple[int, ...]]] = {init}
    frontier: deque[tuple[Node, tuple[int, ...]]] = deque((init,))
    while frontier:
        node, regs = frontier.popleft()
        for val in alphabet:
            if track_counters:
                # Grounding increments counters *before* emitting the
                # valuation, so the determined comparison propositions
                # are computed from the advanced register values.
                regs2 = gadget.step(regs, val)
                full_val = val | gadget.true_props(regs2)
            else:
                regs2, full_val = regs, val
            nxt = _progress(node, full_val)
            if nxt is True:
                return True
            if nxt is False:
                continue
            # ``nxt`` is reachable after >= 1 events: accepting iff its
            # weak end-of-trace collapse discharges every obligation.
            # Checked before the ``seen`` dedup because the *root* enters
            # ``seen`` at depth 0 (empty trace, never finalize-checked);
            # a self-loop like G(a) must still be able to accept here.
            if _finalize(nxt) is True:
                return True
            state = (nxt, regs2)
            if state in seen:
                continue
            if len(seen) >= max_states:
                return None
            seen.add(state)
            frontier.append(state)
    return False
