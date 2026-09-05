"""Formula constructors used by the tests.

Contracts in the library are written as ALTLf formulas; these helpers
build a few recurring shapes so the tests stay readable. They are test
utilities, not part of the package.
"""

from __future__ import annotations

import re as _re

from contragent.formulas.det import DetFormula, _called, _count_var
from contragent.formulas.formula import And, Atom, Const, G, Implies, Le, Not, Or, U, Var


def must_precede(before: str, after: str, desc: str = "") -> DetFormula:
    """``after`` is never called before ``before`` (weak until)."""
    if before == after:
        raise ValueError("must_precede: 'before' and 'after' must differ")
    formula = Or(U(Not(_called(after)), _called(before)), G(Not(_called(after))))
    return DetFormula(
        formula=formula,
        desc=desc or f"{before} must precede {after}",
        kind="must_precede",
        args=(before, after),
    )


def rate_limit(action: str, max_count: int, desc: str = "") -> DetFormula:
    """``action`` is called at most ``max_count`` times."""
    formula = G(Le(_count_var(action), Const(max_count)))
    return DetFormula(
        formula=formula,
        desc=desc or f"{action} limited to {max_count} invocations",
        kind="rate_limit",
        args=(action, max_count),
    )


def no_data_leak(source: str, external: str, desc: str = "") -> DetFormula:
    """Data from ``source`` never flows to ``external``."""
    formula = G(Implies(Atom("contains", source), Not(Atom("flow", source, external))))
    return DetFormula(
        formula=formula,
        desc=desc or f"no data leak from {source} to {external}",
        kind="no_data_leak",
        args=(source, external),
    )


def tool_allowlist(allowed_tools: list[str], desc: str = "") -> DetFormula:
    """Only the listed tools may be called."""
    called_any = Atom("called_any")
    if not allowed_tools:
        formula = G(Not(called_any))
    else:
        allowed = _called(allowed_tools[0])
        for t in allowed_tools[1:]:
            allowed = Or(allowed, _called(t))
        formula = G(Implies(called_any, allowed))
    return DetFormula(
        formula=formula,
        desc=desc or f"only [{', '.join(allowed_tools)}] may be called",
        kind="tool_allowlist",
        args=(tuple(allowed_tools),),
    )


def redirect_to_safe(unsafe: str, safe: str, message: str = "", desc: str = "") -> DetFormula:
    """``unsafe`` is never called; a violation is redirected to ``safe``."""
    from contragent.runtime.strategies import Redirect

    if not unsafe or not unsafe.strip() or not safe or not safe.strip():
        raise ValueError("redirect_to_safe: 'unsafe' and 'safe' must be non-empty tool names")
    if unsafe == safe:
        raise ValueError("redirect_to_safe: 'unsafe' and 'safe' must differ")
    final_desc = desc or f"redirect `{unsafe}` -> `{safe}`" + (f" ({message})" if message else "")
    return DetFormula(
        formula=G(Not(_called(unsafe))),
        desc=final_desc,
        kind="redirect_to_safe",
        args=(unsafe, safe, message),
        enforcement_strategy=Redirect(safe=safe, message=message),
    )


def time_since(predicate_key: str, max_seconds: int | float, desc: str = "") -> DetFormula:
    """``predicate_key`` held within the last ``max_seconds`` seconds."""
    if not isinstance(max_seconds, (int, float)) or max_seconds < 0:
        raise ValueError("time_since: max_seconds must be a non-negative number")
    return DetFormula(
        formula=G(Le(Var("time_since", predicate_key), Const(max_seconds))),
        desc=desc or f"{predicate_key} must have occurred within last {max_seconds}s",
        kind="time_since",
        args=(predicate_key, max_seconds),
    )


def approval_active(action: str, role: str, max_seconds: int | float, desc: str = "") -> DetFormula:
    """``action`` requires a recent ``allow`` decision by ``role``."""
    if not isinstance(max_seconds, (int, float)) or max_seconds < 0:
        raise ValueError("approval_active: max_seconds must be a non-negative number")
    role_key = f"ctx(approval.role, {role})"
    body = And(
        And(
            Atom("ctx_matches", "approval.role", _re.escape(role)),
            Atom("ctx_matches", "approval.decision", "allow"),
        ),
        Le(Var("time_since", role_key), Const(max_seconds)),
    )
    return DetFormula(
        formula=G(Implies(_called(action), body)),
        desc=desc or f"{action} requires active {role} approval (<={max_seconds}s old)",
        kind="approval_active",
        args=(action, role, max_seconds),
    )


def always_followed_by(trigger: str, response: str, desc: str = "") -> DetFormula:
    """Every ``trigger`` is eventually followed by ``response`` (liveness)."""
    if trigger == response:
        raise ValueError("always_followed_by: 'trigger' and 'response' must differ")
    from contragent.formulas.formula import F

    return DetFormula(
        formula=G(Implies(_called(trigger), F(_called(response)))),
        desc=desc or f"{trigger} must always be followed by {response}",
        kind="always_followed_by",
        liveness=True,
        args=(trigger, response),
    )


def no_reversal(commitment: str, contradiction: str, desc: str = "") -> DetFormula:
    """Once ``commitment`` is called, ``contradiction`` never follows."""
    if commitment == contradiction:
        raise ValueError("no_reversal: 'commitment' and 'contradiction' must differ")
    return DetFormula(
        formula=G(Implies(_called(commitment), G(Not(_called(contradiction))))),
        desc=desc or f"{contradiction} must never occur after {commitment}",
        kind="no_reversal",
        args=(commitment, contradiction),
    )


def arg_blacklist(tool: str, param: str, patterns: list[str], desc: str = "") -> DetFormula:
    """Argument ``param`` of ``tool`` matches none of the forbidden regexes."""
    from contragent.formulas.det import _physical_tool

    physical = _physical_tool(tool)
    body = Not(Atom("arg_field_has", physical, param, patterns[0]))
    for pattern in patterns[1:]:
        body = And(body, Not(Atom("arg_field_has", physical, param, pattern)))
    return DetFormula(
        formula=G(Implies(_called(tool), body)),
        desc=desc or f"{tool}.{param} must not match forbidden patterns",
        kind="arg_blacklist",
        args=(tool, param, tuple(patterns)),
    )


def arg_allowlist(tool: str, param: str, patterns: list[str], desc: str = "") -> DetFormula:
    """Argument ``param`` of ``tool`` matches one of the allowed regexes."""
    from contragent.formulas.det import _physical_tool

    if not patterns:
        raise ValueError("arg_allowlist: 'patterns' must be non-empty")
    physical = _physical_tool(tool)
    body = Atom("arg_field_has", physical, param, patterns[0])
    for pattern in patterns[1:]:
        body = Or(body, Atom("arg_field_has", physical, param, pattern))
    return DetFormula(
        formula=G(Implies(_called(tool), body)),
        desc=desc or f"{tool}.{param} must match one of the allowed patterns",
        kind="arg_allowlist",
        args=(tool, param, tuple(patterns)),
    )


def ctx_required(tool: str, key: str, allowed_values: list[str], desc: str = "") -> DetFormula:
    """When ``tool`` is called, context ``key`` holds one of ``allowed_values``."""
    if not allowed_values:
        raise ValueError("ctx_required: 'allowed_values' must not be empty")
    values = [str(v) for v in allowed_values]
    disjunction = Atom("ctx", key, values[0])
    for val in values[1:]:
        disjunction = Or(disjunction, Atom("ctx", key, val))
    return DetFormula(
        formula=G(Implies(_called(tool), disjunction)),
        desc=desc or f"{tool} requires ctx[{key}] in [{', '.join(values)}]",
        kind="ctx_required",
        args=(tool, key, tuple(values)),
    )


def ctx_matches_required(tool: str, key: str, pattern: str, desc: str = "") -> DetFormula:
    """When ``tool`` is called, context ``key`` matches regex ``pattern``."""
    return DetFormula(
        formula=G(Implies(_called(tool), Atom("ctx_matches", key, pattern))),
        desc=desc or f"{tool} requires ctx[{key}] to match /{pattern}/",
        kind="ctx_matches_required",
        args=(tool, key, pattern),
    )
