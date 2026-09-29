"""Numeric coercion for the ordered comparisons of numeric predicates.

A numeric predicate in ContrAgent has the form ``theta(s, a) <bowtie> c``,
where ``theta`` extracts a quantity from the event. Tool arguments reach
the grounder as the integration passes them, and a model routinely writes
an amount as ``"5000"``, ``"$5,000"``, ``"5,000"`` or ``"5000 USD"``. Before
this module, a guard such as ``Not(Gt(ArgValue("pay", "amount"),
Const(1000)))`` reached the evaluator as ``compare("gt", "5000", 1000)``,
which raised ``TypeError`` and fell through to ``False``, so the guard
held and the call was allowed. The extraction ``theta`` is therefore
defined here on every string that unambiguously denotes a number.

Accepted: a plain decimal, float or scientific literal, optionally with
a leading currency symbol, an unambiguous thousands grouping
(``5,000``, ``1,234,567.89``) and a trailing unit or currency code. A
value beyond the float range becomes infinity, so a cap on it fires
rather than being skipped. Refused: ``5,50``, where the comma may be a
decimal separator, and any string that leaves a non-numeric remainder.
A refused value meets the numeric operand once through ``warnings``;
the comparison then evaluates to ``False``, and the missing-arguments
gate of the supervisor is what keeps such a call from running unchecked.
"""

from __future__ import annotations

import re
import warnings

__all__ = ["coerce_ordered", "to_number"]

# Plain decimal, float or scientific literal. Excludes inf, nan, hex and
# the empty string.
_NUMERIC_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")

# Unambiguous thousands grouping: 5,000 / 1,234,567 / 1,234,567.89.
_GROUPED_RE = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")

# A leading currency symbol, optionally after the sign.
_CURRENCY_PREFIX_RE = re.compile(r"^([+-]?)\s*[$€£¥₹]\s*")

# A trailing currency code or unit: "5000 USD", "5000USD", "12 %". No
# ``\s*`` next to the anchor (``\s*X\s*$`` is a polynomial-ReDoS shape);
# whitespace is trimmed separately. The lookbehind keeps the unit from
# eating the tail of a longer word.
_UNIT_SUFFIX_RE = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]{2,4}|%)$")

# Values already reported, so a loop over the same bad argument warns once.
_WARNED: set[str] = set()


def _is_number(v: object) -> bool:
    # bool is excluded: ``True < 2`` already compares natively.
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _normalise(s: str) -> str:
    """Strip the formatting a model wraps around a number."""
    s = s.strip()
    s = _CURRENCY_PREFIX_RE.sub(r"\1", s)
    stripped_unit = _UNIT_SUFFIX_RE.sub("", s).rstrip()
    # Drop the suffix only if a number is left: "USD" alone must stay
    # unparsed, and the exponent in "1e5" must not be taken for a unit.
    if stripped_unit and _GROUPED_RE.match(stripped_unit.replace(" ", "")):
        s = stripped_unit
    elif stripped_unit and _NUMERIC_RE.match(stripped_unit):
        s = stripped_unit
    s = s.strip()
    if _GROUPED_RE.match(s):
        s = s.replace(",", "")
    return s


def to_number(value: object) -> int | float | None:
    """The number ``value`` denotes, or ``None`` when it denotes none.

    Numbers pass through unchanged. A string is normalised as described
    in the module docstring; an integer literal becomes ``int`` and any
    other literal ``float``. Overflow gives infinity.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return None
    s = _normalise(value)
    if not _NUMERIC_RE.match(s):
        return None
    if _INTEGER_RE.match(s):
        return int(s)
    return float(s)


def _warn_incomparable(value: str) -> None:
    key = value[:120]
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(
        f"contragent: a numeric predicate received the non-numeric value "
        f"{value[:60]!r}; the comparison cannot be evaluated and does not "
        "constrain this call. Normalise the argument, or state the rule over "
        "text with arg_field_has.",
        UserWarning,
        stacklevel=4,
    )


def coerce_ordered(left: object, right: object) -> tuple[object, object]:
    """Coerce a numeric-looking string for an ordered comparison.

    Fires only when one operand is a number and the other a string that
    denotes one; everything else (two numbers, two strings, ``None``) is
    returned unchanged. A string that cannot be coerced while the other
    side is numeric is reported once through :mod:`warnings`.
    """
    if _is_number(left) and isinstance(right, str):
        n = to_number(right)
        if n is not None:
            return left, n
        _warn_incomparable(right)
    elif _is_number(right) and isinstance(left, str):
        n = to_number(left)
        if n is not None:
            return n, right
        _warn_incomparable(left)
    return left, right
