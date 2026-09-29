"""``contragent eval`` — batch trace replay with confusion-matrix metrics.

The point of this command is to answer the only question that
matters before flipping ``CONTRAGENT_MODE=gate``:

    "If I turn enforcement on tomorrow, how many *real* incidents do
    my contracts catch, and how much *legit* traffic do they kill?"

Mechanically: replay a labelled corpus of traces against the
configured contracts and emit a per-contract confusion matrix plus
the four headline rates (precision, recall, FPR, FNR).

Labels are read from the filename — files prefixed ``safe_`` are
expected to pass every contract, files prefixed ``unsafe_`` are
expected to be blocked by *at least one* contract.  This convention
is deliberately simple so users can build a corpus by dropping
files into a folder, no schema or front-matter required.

Confusion matrix definitions (per contract):

    TP — predicted block, actually unsafe   (correct catch)
    FP — predicted block, actually safe     (overblock — costs trust)
    FN — predicted allow, actually unsafe   (miss — costs safety)
    TN — predicted allow, actually safe     (correct allow)

For the overall corpus, a trace is "blocked" if *any* contract
violates on it — so a single trigger-happy contract can poison the
whole agent's overblock rate, which is exactly the failure mode
``eval`` is designed to make visible.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Label = Literal["safe", "unsafe", "unknown"]


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """One labelled trace file ready for replay."""

    path: Path
    label: Label
    trace: Any  # contragent.models.trace.Trace — loaded lazily

    @property
    def name(self) -> str:
        return self.path.name


def _label_from_filename(name: str) -> Label:
    """Parse the ``safe_`` / ``unsafe_`` filename prefix.

    Conservative: anything that doesn't match one of the two
    canonical prefixes returns ``"unknown"`` so the eval runner can
    skip it (vs silently mislabelling it as ``safe`` and inflating
    the FPR).
    """
    lower = name.lower()
    if lower.startswith("unsafe_") or lower.startswith("unsafe-"):
        return "unsafe"
    if lower.startswith("safe_") or lower.startswith("safe-"):
        return "safe"
    return "unknown"


def discover_cases(path: Path) -> list[EvalCase]:
    """Walk ``path`` and load every ``*.json`` trace file.

    Accepts either a single file or a directory.  Directory walk is
    NOT recursive — eval corpora are usually shallow and we want
    "dump files into a folder" to be the obvious workflow without
    accidentally sweeping in unrelated ``.json`` from
    ``node_modules/`` and the like.
    """
    from contragent.models.trace import Trace

    files: list[Path]
    if path.is_file():
        files = [path]
    else:
        files = sorted(p for p in path.glob("*.json") if p.is_file())

    cases: list[EvalCase] = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            # Accept BOTH wire formats:
            #   - OTLP export (``resourceSpans``) — what OTel collectors emit.
            #   - ContrAgent-native ``Trace.to_dict()`` (``events``) — what the
            #     library itself serializes; eval must read its own format, or
            #     a native-JSON corpus silently loads as 0-event traces and
            #     every contract passes vacuously (worst-case false-negative).
            if isinstance(data, dict) and "events" in data and "resourceSpans" not in data:
                trace = Trace.from_dict(data)
            else:
                raise ValueError("not a native ContrAgent trace")
        except (json.JSONDecodeError, KeyError, ValueError, AttributeError, TypeError):
            # Skip malformed files rather than abort — the corpus
            # might contain notes or work-in-progress.  ``otel_to_trace``
            # is permissive about shape and can raise a variety of
            # non-Json errors when given the wrong root type (e.g. a
            # list instead of a dict), so we cast a broad-but-bounded
            # net here.
            continue
        cases.append(EvalCase(path=f, label=_label_from_filename(f.name), trace=trace))
    return cases


# ---------------------------------------------------------------------------
# Per-contract result + aggregation
# ---------------------------------------------------------------------------


@dataclass
class CaseOutcome:
    """One (contract × case) verification result."""

    case_name: str
    contract_nl: str
    label: Label
    blocked: bool  # contract said violation
    skipped: bool = False  # unparseable entry — not counted

    @property
    def is_tp(self) -> bool:
        return not self.skipped and self.blocked and self.label == "unsafe"

    @property
    def is_fp(self) -> bool:
        return not self.skipped and self.blocked and self.label == "safe"

    @property
    def is_fn(self) -> bool:
        return not self.skipped and not self.blocked and self.label == "unsafe"

    @property
    def is_tn(self) -> bool:
        return not self.skipped and not self.blocked and self.label == "safe"


@dataclass
class ContractMetrics:
    """Confusion matrix + derived rates for a single contract."""

    contract_nl: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    skipped: int = 0

    @property
    def total_labelled(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    @property
    def fpr(self) -> float | None:
        """Overblock rate among legitimate traffic."""
        denom = self.fp + self.tn
        return self.fp / denom if denom else None

    @property
    def fnr(self) -> float | None:
        """Miss rate among real incidents."""
        denom = self.fn + self.tp
        return self.fn / denom if denom else None


@dataclass
class EvalReport:
    """Full per-contract + overall report."""

    contracts: list[ContractMetrics] = field(default_factory=list)
    n_cases: int = 0
    n_safe: int = 0
    n_unsafe: int = 0
    n_unlabelled: int = 0

    # Overall (any contract blocks → blocked)
    overall_tp: int = 0
    overall_fp: int = 0
    overall_fn: int = 0
    overall_tn: int = 0

    @property
    def overall_fpr(self) -> float | None:
        denom = self.overall_fp + self.overall_tn
        return self.overall_fp / denom if denom else None

    @property
    def overall_fnr(self) -> float | None:
        denom = self.overall_fn + self.overall_tp
        return self.overall_fn / denom if denom else None

    def to_dict(self) -> dict:
        return {
            "n_cases": self.n_cases,
            "n_safe": self.n_safe,
            "n_unsafe": self.n_unsafe,
            "n_unlabelled": self.n_unlabelled,
            "overall": {
                "tp": self.overall_tp,
                "fp": self.overall_fp,
                "fn": self.overall_fn,
                "tn": self.overall_tn,
                "fpr": self.overall_fpr,
                "fnr": self.overall_fnr,
            },
            "contracts": [
                {
                    "nl": m.contract_nl,
                    "tp": m.tp,
                    "fp": m.fp,
                    "fn": m.fn,
                    "tn": m.tn,
                    "skipped": m.skipped,
                    "precision": m.precision,
                    "recall": m.recall,
                    "fpr": m.fpr,
                    "fnr": m.fnr,
                }
                for m in self.contracts
            ],
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _as_det(parsed: Any) -> Any:
    """Normalise a contract entry to a ``DetFormula`` (or ``None``)."""
    from contragent.formulas.det import DetFormula
    from contragent.formulas.formula import FormulaMixin

    if parsed is None:
        return None
    if isinstance(parsed, DetFormula):
        return parsed
    if isinstance(parsed, FormulaMixin):
        return DetFormula(formula=parsed, desc=repr(parsed))
    hard = getattr(parsed, "hard", None)
    if hard is not None:
        return _as_det(hard)
    return None


def resolve_entry(entry: Any) -> tuple[str, Any]:
    """Turn a contract entry (formula text, ``ConstraintEntry``, or formula)
    into ``(label, DetFormula | None)``. Unparseable entries yield ``None``
    and are reported as skipped."""
    from contragent.config import ConfigError, ConstraintEntry, _compile_ltl

    if isinstance(entry, ConstraintEntry):
        if not entry.is_ltl:
            return entry.desc or entry.nl or str(entry), None
        try:
            det = _compile_ltl(entry)
        except ConfigError:
            return entry.desc or entry.ltl, None
        return det.desc, det
    if isinstance(entry, str):
        try:
            det = _compile_ltl(ConstraintEntry(ltl=entry))
        except ConfigError:
            return entry, None
        return det.desc, det
    det = _as_det(entry)
    if det is None:
        return str(entry), None
    return det.desc, det


def _eval_contract_on_trace(parsed: Any, trace: Any) -> bool | None:
    """``True`` if the contract fires (is violated) on the trace, ``False`` if
    it holds, ``None`` if the entry could not be evaluated."""
    det = _as_det(parsed)
    if det is None:
        return None
    from contragent.formulas.evaluator import evaluate as eval_formula
    from contragent.tracer.grounding import collect_content_atoms, ground

    content_atoms = collect_content_atoms([det]) or None
    valuations = ground(trace, content_atoms=content_atoms)
    holds = eval_formula(det.formula, valuations)
    return not holds


def run_eval(cases: Iterable[EvalCase], contracts: list[str]) -> EvalReport:
    """Replay ``cases`` against each NL contract and tally a report.

    Skipped contracts (sto / unparseable) appear in the per-contract
    section with ``skipped > 0`` and zero TP/FP/FN/TN — they're
    visible but excluded from rate calculations because we don't
    deterministically know their predicted outcome.
    """

    report = EvalReport()
    cases = list(cases)
    report.n_cases = len(cases)
    report.n_safe = sum(1 for c in cases if c.label == "safe")
    report.n_unsafe = sum(1 for c in cases if c.label == "unsafe")
    report.n_unlabelled = sum(1 for c in cases if c.label == "unknown")

    # Pre-parse contracts once
    parsed_contracts: list[tuple[str, Any]] = []
    for nl in contracts:
        nl_text, parsed = resolve_entry(nl)
        parsed_contracts.append((nl_text, parsed))
        report.contracts.append(ContractMetrics(contract_nl=nl_text))

    # Per-case outer loop so we can compute "any contract blocked → blocked"
    # for the overall confusion matrix.
    for case in cases:
        any_blocked = False
        for (nl_text, parsed), metric in zip(parsed_contracts, report.contracts, strict=False):
            verdict = _eval_contract_on_trace(parsed, case.trace)
            if verdict is None:
                metric.skipped += 1
                continue
            blocked = bool(verdict)
            any_blocked = any_blocked or blocked
            outcome = CaseOutcome(
                case_name=case.name,
                contract_nl=nl_text,
                label=case.label,
                blocked=blocked,
            )
            if outcome.is_tp:
                metric.tp += 1
            elif outcome.is_fp:
                metric.fp += 1
            elif outcome.is_fn:
                metric.fn += 1
            elif outcome.is_tn:
                metric.tn += 1
            # ``label == "unknown"`` falls through — counted only in
            # ``n_unlabelled``, never in confusion matrix.

        if case.label == "unsafe":
            if any_blocked:
                report.overall_tp += 1
            else:
                report.overall_fn += 1
        elif case.label == "safe":
            if any_blocked:
                report.overall_fp += 1
            else:
                report.overall_tn += 1

    return report


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Baseline diff (CI regression gate)
# ---------------------------------------------------------------------------


def _fmt_delta(d: float | None) -> str:
    """Render a delta as ``+1.5pp`` / ``-0.3pp`` / ``—``.

    Percentage points (not relative %) because that's how operators
    actually reason about FPR: "went from 2% to 4% = +2pp" is the
    intuitive read; "+100%" of a tiny base is misleading.
    """
    if d is None:
        return "    —"
    sign = "+" if d >= 0 else "−"
    return f"{sign}{abs(d) * 100:4.2f}pp"


def _fmt_rate(r: float | None) -> str:
    return "—" if r is None else f"{r * 100:5.1f}%"


def format_report(report: EvalReport) -> str:
    """Human-readable rendering used by the CLI when ``--json`` is off."""
    lines: list[str] = []
    lines.append("")
    lines.append(
        f"Eval — {report.n_cases} cases ({report.n_safe} safe, "
        f"{report.n_unsafe} unsafe, {report.n_unlabelled} unlabelled)"
    )
    lines.append("")

    # Per-contract table
    if report.contracts:
        lines.append("Per contract:")
        header = f"  {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  {'FPR':>6} {'FNR':>6}  {'skip':>4}  contract"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for m in report.contracts:
            nl = m.contract_nl if len(m.contract_nl) <= 60 else m.contract_nl[:57] + "..."
            lines.append(
                f"  {m.tp:>4} {m.fp:>4} {m.fn:>4} {m.tn:>4}  "
                f"{_fmt_rate(m.fpr):>6} {_fmt_rate(m.fnr):>6}  "
                f"{m.skipped:>4}  {nl}"
            )
        lines.append("")

    # Overall
    if report.n_safe + report.n_unsafe > 0:
        lines.append("Overall (any contract blocks → blocked):")
        lines.append(
            f"  TP={report.overall_tp}  FP={report.overall_fp}  "
            f"FN={report.overall_fn}  TN={report.overall_tn}"
        )
        lines.append(f"  FPR (overblock):  {_fmt_rate(report.overall_fpr)}")
        lines.append(f"  FNR (miss):       {_fmt_rate(report.overall_fnr)}")
    else:
        lines.append(
            "No labelled cases — name files ``safe_*.json`` / "
            "``unsafe_*.json`` to enable confusion-matrix metrics."
        )
    lines.append("")
    return "\n".join(lines)
