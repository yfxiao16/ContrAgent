"""Fluent helper for writing contracts in Python.

Example::

    from contragent import ContrAgent, contract
    from contragent.formulas.parser import parse_repr

    guard = ContrAgent(contracts=[
        contract("verify before transfer")
            .assume(parse_repr("F(called('transfer_funds'))"))
            .guarantees(parse_repr("(!(called('transfer_funds')) U called('verify_identity'))"
                                   " | G(!(called('transfer_funds')))")),
    ])
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


def _merge(existing: Any | None, value: Any) -> Any:
    """Append repeated fields while preserving scalar shorthand."""
    if existing is None:
        return value
    if isinstance(existing, list):
        return [*existing, value]
    return [existing, value]


@dataclass(frozen=True)
class ContractBuilder:
    """Builder for one assume-guarantee contract.

    Repeated ``assume`` or ``guarantees`` calls are conjoined, matching
    list-valued ``assumption`` / ``guarantee`` fields.
    """

    desc: str | None = None
    assumption: Any | None = None
    guarantee: Any | None = None

    def assume(self, value: Any) -> ContractBuilder:
        """Add an assumption, the A side of the contract."""
        return replace(self, assumption=_merge(self.assumption, value))

    def guarantees(self, value: Any) -> ContractBuilder:
        """Add a guarantee, the G side of the contract."""
        return replace(self, guarantee=_merge(self.guarantee, value))

    def to_dict(self) -> dict[str, Any]:
        """The contract as the mapping accepted by :class:`~contragent.core.ContrAgent`."""
        if self.guarantee is None:
            raise ValueError("contract(...).guarantees(...) is required")
        out: dict[str, Any] = {"guarantee": self.guarantee}
        if self.desc is not None:
            out["desc"] = self.desc
        if self.assumption is not None:
            out["assumption"] = self.assumption
        return out


def contract(desc: str | None = None) -> ContractBuilder:
    """Start a contract builder."""
    return ContractBuilder(desc=desc)


__all__ = ["ContractBuilder", "contract"]
