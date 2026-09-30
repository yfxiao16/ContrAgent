"""Terminal rendering of a supervised session as a timeline.

``ContrAgent(..., verbose=True)`` prints the session as a vertical
timeline: a header naming the library, then one node per tool call on a
rail down the left. The node says what happened at that point:

* ``●`` the call ran;
* ``⊘`` the call was refused and never ran, with the contract underneath;
* ``◐`` under a node: the call ran but its result was withheld from the
  model, with the assumption that withheld it;
* ``↪`` / ``⏸`` the call was redirected or is held for a human;
* ``■`` the session ended, with the counts and anything still owed.

Colour is used when the stream is a terminal and ``NO_COLOR`` is unset;
``Console(stream, color=False)`` writes plain text anywhere.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, TextIO

__all__ = ["Console"]


def _use_color(stream: TextIO, color: bool | None) -> bool:
    if color is not None:
        return color
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


class _Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _w(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def bold(self, t: str) -> str:
        return self._w("1", t)

    def dim(self, t: str) -> str:
        return self._w("2", t)

    def green(self, t: str) -> str:
        return self._w("32", t)

    def red(self, t: str) -> str:
        return self._w("31", t)

    def yellow(self, t: str) -> str:
        return self._w("33", t)

    def magenta(self, t: str) -> str:
        return self._w("35", t)


def _call(tool: str, args: dict | None) -> str:
    if not args:
        return f"{tool}()"
    inner = ", ".join(f"{k}={json.dumps(v, default=str)}" for k, v in args.items())
    return f"{tool}({inner})"


def _short(text: Any, limit: int = 88) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _reason(result: Any) -> str:
    """The contract descriptions behind a decision."""
    names: list[str] = []
    for v in getattr(result, "violations", []) or []:
        rule = getattr(v, "rule_id", None) or ""
        if rule and rule not in names:
            names.append(rule)
    return "; ".join(names)


class Console:
    def __init__(self, stream: TextIO | None = None, *, color: bool | None = None) -> None:
        self.stream = stream or sys.stdout
        self.s = _Style(_use_color(self.stream, color))

    def _w(self, line: str = "") -> None:
        self.stream.write(line + "\n")

    def _rail(self) -> None:
        self._w(self.s.dim("│"))

    def _note(self, text: str) -> None:
        self._w(f"{self.s.dim('│')}   {text}")

    # ------------------------------------------------------------------
    def banner(
        self, *, library: str, contracts: int, agent_id: str, mode: str, version: str
    ) -> None:
        n = f"{contracts} contract{'s' if contracts != 1 else ''}"
        self._w(self.s.bold(library) + f" · {n} · agent {agent_id} · mode {mode}")

    def call(self, tool: str, args: dict | None, result: Any) -> None:
        """One node for a decided call (after ``guard_before``)."""
        s = self.s
        text = _call(tool, args)
        reason = _short(_reason(result))
        self._rail()
        if result.blocked:
            self._w(f"{s.red('⊘')} {s.bold(text)}")
            self._note(s.red(reason))
        elif result.redirected:
            self._w(f"{s.yellow('↪')} {s.bold(text)}")
            self._note(s.yellow(f"redirected to {result.redirected_to} · {reason}"))
        elif result.escalated:
            self._w(f"{s.yellow('⏸')} {s.bold(text)}")
            self._note(s.yellow(f"held for a human · {reason}"))
        else:
            self._w(f"{s.green('●')} {text}")

    def result(self, tool: str, result: Any) -> None:
        """A note under the last node when its result was withheld
        (after ``guard_after``)."""
        if not result.suppressed:
            return
        self._note(self.s.magenta("◐ result withheld · " + _short(_reason(result))))

    def summary(
        self,
        *,
        calls: int,
        refused: int,
        withheld: int,
        pending: list[str],
        saved: str | None = None,
    ) -> None:
        s = self.s
        self._rail()
        counts = (
            f"{calls} call{'s' if calls != 1 else ''} · {refused} refused · {withheld} withheld"
        )
        if pending:
            owed = f"{len(pending)} obligation{'s' if len(pending) != 1 else ''} pending"
            self._w(f"{s.yellow('■')} {s.bold('session end')} · {counts} · {owed}")
            for desc in pending:
                self._w(f"    {s.yellow('◌')} {_short(desc)}")
        else:
            self._w(f"{s.green('■')} {s.bold('session end')} · {counts} · clean")
        if saved:
            self._w(f"    {s.dim('trace saved to ' + saved)}")
