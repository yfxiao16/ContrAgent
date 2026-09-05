"""Requirement artifacts as inputs to contract formulation.

Two kinds of artifact are loaded here: policy documents (text, Markdown,
PDF), which the formulation pipeline turns into proposed contracts, and
recorded traces, which are replayed offline against a library. Agent
source code is a third input in the paper's pipeline; its analysis lives
outside this artifact.
"""

from __future__ import annotations

from pathlib import Path

from contragent.discovery._types import (
    ConstraintStatus,
    DiscoveryResult,
    DiscoverySource,
    ProposedConstraint,
)
from contragent.discovery.document import DocumentExtractor
from contragent.discovery.loaders import load_documents, load_traces


def discover(
    documents: str | Path | list[str | Path] | None = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    tool_inventory: list[dict] | None = None,
    min_confidence: float = 0.3,
) -> DiscoveryResult:
    """Formulate contracts from policy documents.

    Each document is read and passed through :class:`DocumentExtractor`;
    the returned proposals carry the compiled formula, the model's
    confidence, and the source quote that motivated the contract.
    """
    result = DiscoveryResult()
    if documents is None:
        return result
    paths = documents if isinstance(documents, list) else [documents]
    extractor = DocumentExtractor(
        model=model, api_key=api_key, provider=provider, base_url=base_url
    )
    for text in load_documents([Path(p) for p in paths]):
        result.proposals.extend(
            extractor.extract(text, tool_inventory=tool_inventory, min_confidence=min_confidence)
        )
    return result


__all__ = [
    "discover",
    "DocumentExtractor",
    "DiscoveryResult",
    "DiscoverySource",
    "ConstraintStatus",
    "ProposedConstraint",
    "load_documents",
    "load_traces",
]
