"""Hot-path latency accounting for contract checks.

Every contract check on the online path is timed with
:class:`CheckTimer`; :class:`PerformanceTracker` keeps the samples and
summarizes them as percentiles, overall and per contract. There is no
model call on this path, so the numbers are the full cost of a check.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path


def _percentile(sorted_ns: list[int], pct: float) -> int:
    if not sorted_ns:
        return 0
    idx = min(len(sorted_ns) - 1, max(0, int(round(pct / 100.0 * (len(sorted_ns) - 1)))))
    return sorted_ns[idx]


@dataclass
class BucketStats:
    """Latency statistics (nanoseconds) for a set of checks."""

    n: int = 0
    mean_ns: float = 0.0
    p50_ns: int = 0
    p95_ns: int = 0
    p99_ns: int = 0
    max_ns: int = 0

    @staticmethod
    def from_samples(ns_list: list[int]) -> "BucketStats":
        if not ns_list:
            return BucketStats()
        s = sorted(ns_list)
        return BucketStats(
            n=len(s),
            mean_ns=sum(s) / len(s),
            p50_ns=_percentile(s, 50),
            p95_ns=_percentile(s, 95),
            p99_ns=_percentile(s, 99),
            max_ns=s[-1],
        )

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_us": self.mean_ns / 1000.0,
            "p50_us": self.p50_ns / 1000.0,
            "p95_us": self.p95_ns / 1000.0,
            "p99_us": self.p99_ns / 1000.0,
            "max_us": self.max_ns / 1000.0,
        }


@dataclass
class PerfSummary:
    total_checks: int = 0
    overall: BucketStats = field(default_factory=BucketStats)
    per_contract: dict[str, BucketStats] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_checks": self.total_checks,
            "overall": self.overall.to_dict(),
            "per_contract": {k: v.to_dict() for k, v in self.per_contract.items()},
        }


class PerformanceTracker:
    """Collects per-check latency samples (bounded ring per contract)."""

    def __init__(self, *, ring_size: int = 10_000) -> None:
        self._ring_size = ring_size
        self._all: deque[int] = deque(maxlen=ring_size)
        self._by_label: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=ring_size))
        self._total = 0

    def record(self, label: str, ns: int) -> None:
        self._total += 1
        self._all.append(ns)
        self._by_label[label].append(ns)

    def reset(self) -> None:
        self._all.clear()
        self._by_label.clear()
        self._total = 0

    @property
    def total_checks(self) -> int:
        return self._total

    def summarize(self) -> PerfSummary:
        return PerfSummary(
            total_checks=self._total,
            overall=BucketStats.from_samples(list(self._all)),
            per_contract={k: BucketStats.from_samples(list(v)) for k, v in self._by_label.items()},
        )

    def export_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.summarize().to_dict(), indent=2))
        return p


class CheckTimer:
    """Context manager timing one contract check into a tracker (or nothing)."""

    def __init__(self, tracker: PerformanceTracker | None, label: str) -> None:
        self._tracker = tracker
        self._label = label
        self._start = 0

    def __enter__(self) -> "CheckTimer":
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tracker is not None:
            self._tracker.record(self._label, time.perf_counter_ns() - self._start)


def format_summary(summary: PerfSummary) -> str:
    o = summary.overall
    lines = [
        f"contract checks: {summary.total_checks}",
        f"  p50 {o.p50_ns / 1000:.1f} us   p95 {o.p95_ns / 1000:.1f} us   "
        f"p99 {o.p99_ns / 1000:.1f} us   max {o.max_ns / 1000:.1f} us",
    ]
    for label, b in sorted(summary.per_contract.items(), key=lambda kv: -kv[1].p99_ns)[:10]:
        lines.append(f"  {label[:60]:<60s} n={b.n:<6d} p99 {b.p99_ns / 1000:.1f} us")
    return "\n".join(lines)


__all__ = ["PerformanceTracker", "PerfSummary", "BucketStats", "CheckTimer", "format_summary"]
