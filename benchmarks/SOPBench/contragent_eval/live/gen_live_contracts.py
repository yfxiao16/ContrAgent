"""
Generate the live-only contract overlay (contracts/<domain>.live.yaml).

These contracts use ``state_user_exists`` -- an atom ContrAgent observes from the
REAL world state at decision time (the acting user's row is present in the DB).
The offline static proxy could only see that an existence check was *called*;
in-loop the guard reads the actual table, so it can require the stronger
"the user actually exists" (or, for account creation, "does not yet exist").

Scoped by per-task flags grounded in enforce.py:
  gate_user_must_exist      -> the task's SOP requires the acting user to exist
  gate_user_must_not_exist  -> the task's SOP requires them NOT to exist yet
so each rule is inert on tasks/goals whose SOP has no existence gate.

Run:  python live/gen_live_contracts.py <domain> [<domain> ...]
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.dirname(_HERE)
_SOP_ROOT = os.path.dirname(_EVAL_DIR)
for p in (_SOP_ROOT, os.path.dirname(os.path.dirname(_SOP_ROOT))):
    if p not in sys.path:
        sys.path.insert(0, p)

from live import env_loader as el  # noqa: E402

SKIP_PREFIX = ("internal_", "login_", "logout_", "authenticate_", "get_", "evaluation_")


def goals_for(domain):
    _, asst = el.load_modules(domain)
    return [a["name"] for a in asst.actions if not a["name"].startswith(SKIP_PREFIX)]


def gen(domain):
    lines = [
        "# AUTO-GENERATED live-only overlay (see live/gen_live_contracts.py).",
        "# Uses state_user_exists, observed from the real world state in-loop.",
        f"# Loaded IN ADDITION to contracts/{domain}.yaml by the enforce guard only;",
        "# the offline detection config is unchanged.",
        "agents:",
        '  "*":',
        "    contracts:",
    ]
    for g in goals_for(domain):
        lines.append(
            f'      - desc: "{g} requires the acting user to exist (live existence)"'
        )
        lines.append(
            f'        G: {{ltl: "G((called({g}) & arg_value({g}, succeeded) >= 1 '
            f"& arg_value({g}, gate_user_must_exist) >= 1) -> "
            f'arg_value({g}, state_user_exists) >= 1)"}}'
        )
        lines.append(
            f'      - desc: "{g} requires the acting user to NOT already exist (live existence)"'
        )
        lines.append(
            f'        G: {{ltl: "G((called({g}) & arg_value({g}, succeeded) >= 1 '
            f"& arg_value({g}, gate_user_must_not_exist) >= 1) -> "
            f'arg_value({g}, state_user_exists) <= 0)"}}'
        )
    out = os.path.join(_EVAL_DIR, "contracts", f"{domain}.live.yaml")
    open(out, "w").write("\n".join(lines) + "\n")
    print("wrote", out, f"({len(goals_for(domain))} goals)")


if __name__ == "__main__":
    for d in sys.argv[1:] or list(el.DOMAIN_CLASSES):
        gen(d)
