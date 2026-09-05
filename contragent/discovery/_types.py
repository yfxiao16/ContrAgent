"""Shared types for contracts proposed from requirement artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from contragent.formulas.det import DetFormula


class DiscoverySource(str, Enum):
    """Where a proposed contract came from."""

    BUILTIN = "builtin"
    USER_DEFINED = "user_defined"
    AUTO_EXTRACTED = "auto_extracted"


class ConstraintStatus(str, Enum):
    """Review status of a proposed contract."""

    PROPOSED = "proposed"
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass
class ProposedConstraint:
    """A contract candidate produced by the formulation pipeline.

    Attributes:
        formula: The compiled guarantee with its description.
        assumption: The compiled assumption, if the requirement was conditional.
        source: Where the constraint originated.
        extractor: Which extractor produced it (``"document"``, ``"nl"``).
        confidence: The model's confidence in [0, 1].
        status: Review status.
        provenance: The passage or file the constraint was derived from.
        nl_description: Natural-language reading of the formula.
        evidence: Extractor-specific supporting data.
        validation_errors: Errors found during validation.
    """

    formula: DetFormula | None = None
    assumption: DetFormula | None = None
    source: DiscoverySource = DiscoverySource.AUTO_EXTRACTED
    extractor: str = ""
    confidence: float = 1.0
    status: ConstraintStatus = ConstraintStatus.PROPOSED
    provenance: str = ""
    nl_description: str = ""
    evidence: dict = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.validation_errors) == 0 and self.formula is not None


@dataclass
class DiscoveryResult:
    """All contracts proposed from one set of requirement artifacts."""

    proposals: list[ProposedConstraint] = field(default_factory=list)

    @property
    def accepted(self) -> list[ProposedConstraint]:
        return [p for p in self.proposals if p.ok]
