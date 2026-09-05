"""Adapter for the ``mus2muc`` LTLf MUC enumerator (Ielo et al., AAAI 2026).

``mus2muc`` (https://github.com/ainnoot/mus2muc) enumerates **all**
minimal unsatisfiable cores of an LTLf specification in conjunctive
form, via ASP minimal-unsatisfiable-subprogram enumeration (patched
WASP) certified by an external LTLf solver (``aaltaf`` or ``black``).
When its toolchain is installed, :func:`contragent.analysis.conflicts.
check_conflicts` uses it for step 1 (core extraction) instead of the
native deletion-based shrink, upgrading the report from "disjoint
cores" to an exhaustive enumeration.

Requirements (all external — not installable from PyPI):

* the ``mus2muc`` Python package: ``pip install
  git+https://github.com/ainnoot/mus2muc``
* the patched ``wasp`` solver and an ``aaltaf`` or ``black``
  executable in one folder — pass ``bin_folder=``, set
  ``$CONTRAGENT_MUS2MUC_BIN``, or install them into ``/usr/bin``
  (mus2muc's default). See the mus2muc README for build/patch
  instructions.

Serialization reuses the canonical leaf abstraction from
:mod:`contragent.formulas.sat`, so both backends agree on which
propositions exist and on complement mapping. Grounding-derived domain
constraints are emitted as extra ``dom<i>`` conjuncts and filtered out
of the returned cores.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

from contragent.formulas.sat import _abstract, _LeafTable, Node

# Exit codes from mus2muc.enumeration.MUS2MUCResult that indicate a
# completed (or deliberately truncated) enumeration rather than a crash.
_OK_EXIT_CODES = {0, 10, 20, 30}  # ok / FOUND_ALL_MUCS / TIMEOUT / FOUND_REQUIRED


class Mus2mucUnavailable(RuntimeError):
    """The mus2muc toolchain is not installed or not fully set up."""


# ---------------------------------------------------------------------------
# Availability probing
# ---------------------------------------------------------------------------


def _resolve_bin_folder(bin_folder: str | Path | None) -> Path:
    if bin_folder is not None:
        return Path(bin_folder)
    env = os.environ.get("CONTRAGENT_MUS2MUC_BIN")
    if env:
        return Path(env)
    return Path("/usr/bin")


def _package_present() -> bool:
    try:
        return importlib.util.find_spec("mus2muc") is not None
    except (ImportError, ValueError):
        return False


def _certifier_in(folder: Path) -> str | None:
    for name in ("aaltaf", "black"):
        if (folder / name).is_file():
            return name
    return None


def unavailable_reason(*, bin_folder: str | Path | None = None) -> str:
    """Explain what is missing, or return '' if everything is in place."""
    missing: list[str] = []
    if not _package_present():
        missing.append(
            "the mus2muc package (pip install git+https://github.com/ainnoot/mus2muc)"
        )
    folder = _resolve_bin_folder(bin_folder)
    if not (folder / "wasp").is_file():
        missing.append(f"the patched wasp executable in {folder}")
    if _certifier_in(folder) is None:
        missing.append(f"an aaltaf or black executable in {folder}")
    if not missing:
        return ""
    return (
        "mus2muc backend unavailable — missing: "
        + "; ".join(missing)
        + ". See https://github.com/ainnoot/mus2muc for setup, or use "
        "backend='native'."
    )


def is_available(*, bin_folder: str | Path | None = None) -> bool:
    """True when the mus2muc package and its executables are all present."""
    return unavailable_reason(bin_folder=bin_folder) == ""


# ---------------------------------------------------------------------------
# Serialization to the .ltlfconj input format
# ---------------------------------------------------------------------------


def _node_to_str(node: Node) -> str:
    """Render a canonical sat-node in mus2muc's LTLf grammar.

    Propositions are emitted as generated lowercase symbols ``p<i>``
    (atom keys like ``called(x)`` contain characters the grammar cannot
    quote). ``X`` is emitted as ``WX`` because ContrAgent's runtime
    ``X`` has weak-next semantics at trace end; ``G``/``F``/``U``
    coincide with standard LTLf on non-empty finite traces.
    """
    if node is True:
        # No boolean constants in the grammar; (p0 | !p0) is valid but
        # constants never survive _mk_and/_mk_or canonicalization.
        raise ValueError("cannot serialize constant True")
    if node is False:
        raise ValueError("cannot serialize constant False")
    tag = node[0]
    if tag == "lit":
        i = node[1]
        return f"p{i}" if i > 0 else f"!p{-i}"
    if tag == "and":
        return "(" + " & ".join(sorted(_node_to_str(x) for x in node[1])) + ")"
    if tag == "or":
        return "(" + " | ".join(sorted(_node_to_str(x) for x in node[1])) + ")"
    if tag == "not":
        return f"!({_node_to_str(node[1])})"
    if tag == "G":
        return f"G({_node_to_str(node[1])})"
    if tag == "F":
        return f"F({_node_to_str(node[1])})"
    if tag == "X":
        return f"WX({_node_to_str(node[1])})"
    if tag == "U":
        return f"({_node_to_str(node[1])} U {_node_to_str(node[2])})"
    raise TypeError(f"unknown node tag {tag!r}")  # pragma: no cover


def _theory_dom_bodies(table, checker) -> list[str]:
    """Pointwise theory facts, propositionalized for the LTLf solver.

    mus2muc reasons purely propositionally, so the arithmetic relations
    the native search gets from its per-valuation theory filter are
    re-encoded here as ``G(...)`` conjuncts over unary and binary
    comparison-proposition combinations (impossible comparisons,
    mutually exclusive pairs, and implication chains between constants
    over one register). Higher-arity theory interactions are not
    encoded — the native backend remains the reference for those.
    """
    if checker is None or not table.cmp_info:
        return []
    from contragent.formulas.theory import TheoryLiteral

    def lit(pid: int, positive: bool) -> TheoryLiteral:
        op, left, right, total = table.cmp_info[pid]
        return TheoryLiteral(
            op=op, left=left, right=right, total=total, positive=positive
        )

    def ok(lits) -> bool:
        try:
            return checker.consistent(lits)
        except Exception:
            return True  # broken checker must not manufacture constraints

    out: list[str] = []
    infos = sorted(table.cmp_info.items())
    for i, (pi, info_i) in enumerate(infos):
        if not ok([lit(pi, True)]):
            out.append(f"G(!p{pi})")  # e.g. count <= -1 can never hold
        if info_i[3] and not ok([lit(pi, False)]):
            out.append(f"G(p{pi})")
        for pj, info_j in infos[i + 1 :]:
            if not ok([lit(pi, True), lit(pj, True)]):
                out.append(f"G(!(p{pi} & p{pj}))")
            # A negative literal only carries meaning for total terms.
            if info_j[3] and not ok([lit(pi, True), lit(pj, False)]):
                out.append(f"G(p{pi} -> p{pj})")
            if info_i[3] and not ok([lit(pi, False), lit(pj, True)]):
                out.append(f"G(p{pj} -> p{pi})")
    return out


def units_to_ltlfconj(
    units: Sequence,
    *,
    mutex_groups: Iterable[Iterable[str]] = (),
    implications: Iterable[tuple[str, str]] = (),
    extra_formulas: Iterable = (),
    theory_checker=None,
) -> tuple[str, dict[str, object]]:
    """Serialize contract units (+ domain constraints) to ``.ltlfconj``.

    Returns the file text and a mapping from conjunct identifier
    (``c<i>``) to the unit it labels. Domain-constraint conjuncts
    (mutex/implication encodings, ``extra_formulas`` axioms, and the
    propositionalized pointwise theory facts from ``theory_checker``)
    are labeled ``dom<i>`` and are absent from the mapping so callers
    can filter them from reported cores.
    """
    table = _LeafTable()
    lines: list[str] = []
    id_to_unit: dict[str, object] = {}

    for n, unit in enumerate(units):
        node = _abstract(unit.combined, table)
        if node is True:
            continue  # a vacuously-true conjunct can never be in a core
        cid = f"c{n}"
        if node is False:
            # Self-contradictory even propositionally; emit an
            # unsatisfiable conjunct so it surfaces as a size-1 core.
            body = "(p1 & !p1)" if table.prop_ids else "(q & !q)"
        else:
            body = _node_to_str(node)
        lines.append(f"{cid} := {body};")
        id_to_unit[cid] = unit

    dom_bodies: list[str] = []
    for group in mutex_groups:
        ids = [table.atom_ids[k] for k in group if k in table.atom_ids]
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                dom_bodies.append(f"G(!(p{a} & p{b}))")
    for ante, cons in implications:
        if ante in table.atom_ids and cons in table.atom_ids:
            dom_bodies.append(f"G(p{table.atom_ids[ante]} -> p{table.atom_ids[cons]})")
    for axiom in extra_formulas:
        node = _abstract(axiom, table)
        if node is True:
            continue
        dom_bodies.append(_node_to_str(node))
    dom_bodies.extend(_theory_dom_bodies(table, theory_checker))
    for n, body in enumerate(dom_bodies):
        lines.append(f"dom{n} := {body};")

    return "\n".join(lines) + "\n", id_to_unit


# ---------------------------------------------------------------------------
# Enumeration via the mus2muc CLI
# ---------------------------------------------------------------------------


def _mus2muc_argv() -> list[str]:
    exe = shutil.which("mus2muc")
    if exe:
        return [exe]
    # Package importable but no console script on PATH (e.g. installed
    # into a different env layout): run its argparse main directly.
    return [sys.executable, "-c", "from mus2muc.cli import main; main()"]


def enumerate_mucs(
    units: Sequence,
    *,
    mutex_groups: Iterable[Iterable[str]] = (),
    implications: Iterable[tuple[str, str]] = (),
    extra_formulas: Iterable = (),
    theory_checker=None,
    bin_folder: str | Path | None = None,
    timeout: int = 60,
    max_mucs: int | None = None,
) -> list[list]:
    """Enumerate all minimal unsatisfiable cores of the units' conjunction.

    Runs the ``mus2muc`` solver and maps each reported core back to
    :class:`~contragent.analysis.conflicts.ContractUnit` lists, with
    domain-constraint conjuncts stripped and duplicate contract sets
    (cores differing only in domain conjuncts) deduplicated.

    Raises:
        Mus2mucUnavailable: If the toolchain is missing or the solver
            fails outright.
    """
    reason = unavailable_reason(bin_folder=bin_folder)
    if reason:
        raise Mus2mucUnavailable(reason)

    folder = _resolve_bin_folder(bin_folder)
    certifier = _certifier_in(folder)
    text, id_to_unit = units_to_ltlfconj(
        units,
        mutex_groups=mutex_groups,
        implications=implications,
        extra_formulas=extra_formulas,
        theory_checker=theory_checker,
    )
    if not id_to_unit:
        return []

    with tempfile.TemporaryDirectory(prefix="contragent-mus2muc-") as tmp:
        spec_path = Path(tmp) / "library.ltlfconj"
        out_path = Path(tmp) / "mucs.jsonl"
        spec_path.write_text(text)
        argv = _mus2muc_argv() + [
            str(spec_path),
            "--bin-folder",
            str(folder),
            "--certifier",
            certifier or "aaltaf",
            "-t",
            str(timeout),
            "-o",
            str(out_path),
        ]
        if max_mucs is not None:
            argv += ["-n", str(max_mucs)]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout + 30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise Mus2mucUnavailable(f"mus2muc invocation failed: {exc}") from exc
        if proc.returncode not in _OK_EXIT_CODES:
            raise Mus2mucUnavailable(
                f"mus2muc exited with code {proc.returncode}: "
                f"{proc.stderr.strip()[:500]}"
            )
        raw = out_path.read_text() if out_path.exists() else proc.stdout

    cores: list[list] = []
    seen: set[frozenset[str]] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        conjuncts = [c for c in item.get("conjuncts", ()) if c in id_to_unit]
        if not conjuncts:
            continue
        key = frozenset(conjuncts)
        if key in seen:
            continue
        seen.add(key)
        cores.append([id_to_unit[c] for c in conjuncts])
    return cores
