"""Prototype: compile a SOPBench constraints tree into an LTLf guarantee.

SOPBench semantics (env/dep_eval.py): chain/and = conjunction (all must hold),
gate/or = disjunction (any holds). A `single` predicate maps to an observable
fact already grounded on the goal call. We compile the tree to one LTL formula
that is TRUE iff the (observable part of the) SOP is satisfied; the enforce
guarantee `G(goal & succeeded -> <tree>)` then fires exactly on a completed goal
whose SOP is violated.

Honesty: we compile from the public constraints tree (the SOP spec) and evaluate
predicates against DB-derived facts -- never the action_should_succeed label. An
unobservable branch makes its enclosing OR conservatively "satisfiable" (we
cannot prove a violation through it), so coverage is honest and FP-free.
"""
from __future__ import annotations

# predicate -> expression that is TRUE iff the predicate holds, over atoms the
# grounder already emits on a book_room call.
FRAG = {
    "room_type_available_for_dates": "arg_value({goal}, rooms_available) >= 1",
    "internal_valid_room_type": "arg_value({goal}, state_room_type_valid) >= 1",
    "valid_booking_date_pair": "arg_value({goal}, state_valid_date_pair) >= 1",
    "amount_positive_restr": "arg_value({goal}, state_amount_positive) >= 1",
    "sufficient_amount_for_booking":
        "arg_value({goal}, amount) >= arg_value({goal}, booking_total_fee)",
    # has_exceeded_maximum_stays is TRUE iff the stay exceeds the cap, i.e. NOT within.
    "has_exceeded_maximum_stays": "arg_value({goal}, state_within_max_stays) < 1",
    "internal_is_loyalty_member": "arg_value({goal}, state_is_loyalty_member) >= 1",
    "is_gold_or_higher_member": "arg_value({goal}, state_is_gold_plus) >= 1",
    "is_booking_date_within_lead_time_range":
        "(arg_value({goal}, now_epoch) >= arg_value({goal}, lead_lower_epoch) "
        "& arg_value({goal}, now_epoch) <= arg_value({goal}, lead_upper_epoch))",
    "has_overlapping_booking_for_booking":
        "arg_value({goal}, state_has_overlapping) >= 1",
    # --- healthcare (atoms already grounded by convert.py) ---
    "provider_available": "arg_value({goal}, provider_available) >= 1",
    "provider_authorized": "arg_value({goal}, provider_authorized) >= 1",
    "provider_covers_policy": "arg_value({goal}, provider_covers) >= 1",
    "internal_check_provider_exists": "arg_value({goal}, state_provider_exists) >= 1",
    "policy_active": "arg_value({goal}, state_policy_active) >= 1",
    "appointment_date_valid":
        "arg_value({goal}, appt_epoch) >= arg_value({goal}, now_epoch)",
    "provider_not_already_authorized":
        "arg_value({goal}, provider_not_authorized) >= 1",
    # --- library (atoms grounded by convert.py; see borrow_book grounder) ---
    "within_borrow_limit": "arg_value({goal}, state_within_borrow_limit) >= 1",
    "user_book_not_borrowed": "arg_value({goal}, state_user_book_not_borrowed) >= 1",
    "internal_check_book_available": "arg_value({goal}, state_book_available) >= 1",
    "valid_membership": "arg_value({goal}, state_valid_membership) >= 1",
    # internal_is_restricted is TRUE iff the book is restricted; the SOP usually
    # requires "not restricted", so a restricted target violates it.
    "internal_is_restricted": "arg_value({goal}, state_book_restricted) >= 1",
    "internal_check_book_exist": "arg_value({goal}, state_book_exists) >= 1",
    "internal_check_username_exist": "arg_value({goal}, state_user_exists) >= 1",
    # --- university (course facts already grounded by convert.py) ---
    "internal_check_course_exists": "arg_value({goal}, state_course_exists) >= 1",
    "internal_check_username_exists": "arg_value({goal}, state_user_exists) >= 1",
    "course_has_capacity": "arg_value({goal}, state_course_has_capacity) >= 1",
    "course_not_completed": "arg_value({goal}, course_not_completed) >= 1",
    "not_on_probation": "arg_value({goal}, state_not_on_probation) >= 1",
    "valid_residency_status": "arg_value({goal}, state_residency_valid) >= 1",
    "tuition_balance_zero": "arg_value({goal}, state_tuition_balance) <= 0",
}

_FLIP = {">=": "<", "<=": ">", ">": "<=", "<": ">="}


def _neg(expr):
    # only single-comparison fragments are negatable safely
    if expr is None or " & " in expr or " | " in expr:
        return None  # compound: cannot flip safely -> treat as unobservable
    for op, inv in _FLIP.items():
        if f" {op} " in expr:
            return expr.replace(f" {op} ", f" {inv} ", 1)
    return None


def compile_node(node):
    """Return an LTL string TRUE iff node's SOP holds, or None if unobservable."""
    if not isinstance(node, list) or not node:
        return None
    op = node[0]
    if op == "single":
        name = node[1]
        if isinstance(name, str) and name.startswith("not "):
            base = name[4:]
            frag = FRAG.get(base)
            return _neg(frag) if frag else None
        return FRAG.get(name)
    if op == "not":
        inner = compile_node(node[1])
        return _neg(inner) if inner else None
    if op in ("and", "chain"):
        parts = [compile_node(c) for c in node[1]]
        parts = [p for p in parts if p]
        return "(" + " & ".join(parts) + ")" if parts else None
    if op in ("or", "gate"):
        parts = [compile_node(c) for c in node[1]]
        # any unobservable branch => OR could be satisfied through it => cannot
        # prove a violation => conservatively unobservable (FP-free).
        if any(p is None for p in parts):
            return None
        return "(" + " | ".join(parts) + ")" if parts else None
    return None


def compile_guarantee(goal, tree):
    body = compile_node(tree)
    if not body:
        return None
    formula = f"G((called({goal}) & arg_value({goal}, succeeded) >= 1) -> {body})"
    return formula.replace("{goal}", goal)
