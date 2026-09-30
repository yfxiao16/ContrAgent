"""Supervise a tool-calling agent loop.

    python3 examples/agent_loop.py

This is the shape of every integration: the loop you already have, with
the tools wrapped by ``guard.wrap`` so each call passes ``guard_before``
on the way in and ``guard_after`` on the way out. A refused call never
reaches the tool; the wrapper returns the refusal text, which goes back
to the model as the tool result, and the model chooses again. The model
here is a script, so the run is the same every time and needs no
credentials.

The contracts are written with the fluent helper instead of YAML; the
two forms compile to the same monitors.
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from contragent import ContrAgent, contract, parse_repr  # noqa: E402

# ---------------------------------------------------------------------------
# Tools: what the agent can do. Any callables; the supervisor never sees them.
# ---------------------------------------------------------------------------

ACCOUNT = {"verified": False, "balance": 900.0, "receipts": []}


def verify_identity(customer_id: str) -> dict:
    ACCOUNT["verified"] = True
    return {"ok": True, "customer_id": customer_id}


def transfer_funds(amount: float, to: str) -> dict:
    ACCOUNT["balance"] -= amount
    return {"ok": True, "balance": ACCOUNT["balance"]}


def send_receipt(to: str) -> dict:
    ACCOUNT["receipts"].append(to)
    return {"ok": True}


TOOLS = {
    "verify_identity": verify_identity,
    "transfer_funds": transfer_funds,
    "send_receipt": send_receipt,
}

# ---------------------------------------------------------------------------
# Contracts: what is allowed and required, kept apart from the tools.
# ---------------------------------------------------------------------------

LIBRARY = [
    contract("identity must be verified before funds move").guarantees(
        parse_repr(
            "(!(called('transfer_funds')) U called('verify_identity')) | G(!(called('transfer_funds')))"
        )
    ),
    contract("at most one transfer per session").guarantees(
        parse_repr("G(Var('count', 'transfer_funds') <= 1)")
    ),
    contract("every transfer is eventually receipted").guarantees(
        parse_repr("G(called('transfer_funds') -> F(called('send_receipt')))")
    ),
]

# ---------------------------------------------------------------------------
# The model: scripted here. A real one returns the next tool call given the
# conversation so far; the refusal feedback is what steers it.
# ---------------------------------------------------------------------------

SCRIPT = iter(
    [
        ("transfer_funds", {"amount": 250.0, "to": "ACME"}),  # too early: refused
        ("verify_identity", {"customer_id": "C-1024"}),
        ("transfer_funds", {"amount": 250.0, "to": "ACME"}),  # now allowed
        ("transfer_funds", {"amount": 250.0, "to": "ACME"}),  # second one: refused
        ("send_receipt", {"to": "C-1024"}),
        None,  # the model ends its turn
    ]
)


def model(messages: list[dict]) -> tuple[str, dict] | None:
    return next(SCRIPT)


# ---------------------------------------------------------------------------
# The loop. Two lines of supervision around each call.
# ---------------------------------------------------------------------------

guard = ContrAgent(agent_id="bank_agent", contracts=LIBRARY)
tools = guard.wrap(TOOLS)  # every call now passes guard_before and guard_after
messages: list[dict] = [{"role": "user", "content": "Send $250 to ACME for customer C-1024."}]

while True:
    proposal = model(messages)
    if proposal is None:
        break
    name, args = proposal

    output = tools[name](**args)  # a refused call returns the refusal text instead
    if isinstance(output, str):
        print(f"REFUSED  {name}({args})\n         {output}")
    else:
        print(f"ran      {name}({args}) -> {output}")
    messages.append({"role": "tool", "name": name, "content": str(output)})

pending = guard.finish_session()  # eventualities still owed are violations
print("\nsession end:", "clean" if not pending else [v.desc for v in pending])
print("account:", ACCOUNT)
