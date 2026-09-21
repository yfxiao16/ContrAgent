"""One spelling of a tool name, shared by contracts and grounding.

The ``Call(T)`` predicate of the paper asks whether tool ``T`` is invoked.
In the implementation a predicate key is a dictionary key, so
``called(issue_refund)`` and ``called(Issue_Refund)`` used to be two
unrelated entries and a contract only ever looked up the spelling its
author typed. A rule written against ``issue_refund`` was then silently
inert on an event whose tool arrived as ``Issue_Refund``, as
``issue_refund `` with a stray space, or as ``mcp__finance__issue_refund``,
the documented MCP wire format. Since the shipped contracts are of the
shapes ``G(premise -> conclusion)`` and ``Or(U(!after, before),
G(!after))``, a name that never matches makes them read as satisfied
while the guarded call runs unchecked.

Two functions, used on opposite sides of the same join:

* :func:`canonical_tool` is what a *contract* keys on. The formula
  constructors apply it once, at construction, so a contract keys on a
  single spelling.
* :func:`tool_aliases` is what an *event* answers to. A tool call is a
  fact and is not rewritten; it is grounded under every spelling a
  contract might reasonably have used, canonical form included.

Widening only ever makes more contracts apply to a call, never fewer, so
the failure this introduces is a rule firing on a tool whose name merely
resembles the one named, which is visible and reportable, unlike the
silent pass it replaces.
"""

from __future__ import annotations

import re

__all__ = ["canonical_tool", "tool_aliases", "MCP_TOOL_RE"]

# MCP wire format: ``mcp__<server>__<tool>``. The server segment has no
# ``__`` of its own; the tool segment may, so the split is on the first
# ``__`` after the prefix and the remainder is the tool.
MCP_TOOL_RE = re.compile(r"^mcp__(?P<server>[^_]+(?:_[^_]+)*)__(?P<tool>.+)$")


def _strip_mcp(name: str) -> str | None:
    """``mcp__finance__issue_refund`` -> ``issue_refund``; else ``None``."""
    m = MCP_TOOL_RE.match(name)
    return m.group("tool") if m else None


def canonical_tool(tool: str) -> str:
    """The single spelling a contract keys on.

    Strips surrounding whitespace and case-folds. The MCP prefix is not
    stripped: a contract that names ``mcp__finance__issue_refund`` means
    that server's tool and must not widen to every tool of that name.
    Grounding supplies the bare-name alias so the other direction joins.
    """
    return str(tool).strip().casefold()


def tool_aliases(tool: str) -> tuple[str, ...]:
    """Every spelling an event's tool call answers to.

    Ordered and duplicate-free. The raw name comes first so existing
    traces and predicate keys keep working unchanged.
    """
    raw = str(tool)
    out = [raw]
    stripped = raw.strip()
    out.append(stripped)
    out.append(stripped.casefold())
    bare = _strip_mcp(stripped)
    if bare:
        out.append(bare)
        out.append(bare.casefold())
    return tuple(k for k in dict.fromkeys(out) if k)
