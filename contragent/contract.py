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
    activate_at: str | None = None
    assumption_mode: str | None = None

    def assume(self, value: Any) -> ContractBuilder:
        """Add an assumption, the A side of the contract."""
        return replace(self, assumption=_merge(self.assumption, value))

    def guarantees(self, value: Any) -> ContractBuilder:
        """Add a guarantee, the G side of the contract."""
        return replace(self, guarantee=_merge(self.guarantee, value))

    def from_first_match(self) -> ContractBuilder:
        """Check the guarantee only from the event that fires the assumption."""
        return replace(self, activate_at="first_match")

    def enforce_assumption(self) -> ContractBuilder:
        """Restrict the environment to satisfy the assumption.

        The supervisor suppresses a return or input event that would
        falsify the assumption instead of only evaluating it. Without
        this call the assumption is monitored.
        """
        return replace(self, assumption_mode="enforced")

    def to_dict(self) -> dict[str, Any]:
        """The contract as the mapping accepted by :class:`~contragent.core.ContrAgent`."""
        if self.guarantee is None:
            raise ValueError("contract(...).guarantees(...) is required")
        out: dict[str, Any] = {"guarantee": self.guarantee}
        if self.desc is not None:
            out["desc"] = self.desc
        if self.assumption is not None:
            out["assumption"] = self.assumption
        if self.activate_at is not None:
            out["activate_at"] = self.activate_at
        if self.assumption_mode is not None:
            out["assumption_mode"] = self.assumption_mode
        return out


def contract(desc: str | None = None) -> ContractBuilder:
    """Start a contract builder."""
    return ContractBuilder(desc=desc)


__all__ = ["ContractBuilder", "contract"]
