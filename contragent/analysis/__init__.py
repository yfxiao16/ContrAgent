"""Static (load-time) analyses over a contract library.

Currently one analysis: the conflict-freedom check, :func:`check_conflicts`.
"""

from contragent.analysis.conflicts import (
    ConflictCore,
    ConflictReport,
    check_conflicts,
    extract_muc,
)

__all__ = [
    "ConflictCore",
    "ConflictReport",
    "check_conflicts",
    "extract_muc",
]
