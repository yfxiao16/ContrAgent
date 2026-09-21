"""Re-run specific tasks in enforce mode and dump the full call sequence,
grounded goal args, and which contracts fired -- for failure forensics.

Usage:
  PYTHONPATH=contragent_eval:.:../.. python live/debug_cases.py bank 53 54 68 73 6 78
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.dirname(_HERE)
_SOP_ROOT = os.path.dirname(_EVAL_DIR)
for p in (_SOP_ROOT, os.path.dirname(os.path.dirname(_SOP_ROOT))):
    if p not in sys.path:
        sys.path.insert(0, p)

from live import env_loader as el  # noqa: E402
from live import sop as sopmod  # noqa: E402
from live.agent import run_agent  # noqa: E402
from live.enforce import LiveEnforcer  # noqa: E402
from live.run import base_system, load_tasks  # noqa: E402

GOAL_ARG_KEYS = (
    "state_user_exists",
    "gate_user_must_exist",
    "gate_user_must_not_exist",
    "prior_logged_in",
    "prior_authenticated_admin",
    "amount_dollars",
    "state_currency_available",
    "state_credit_score",
    "state_owed_balance",
    "state_balance",
)


def main():
    domain = sys.argv[1]
    idxs = [int(x) for x in sys.argv[2:]]
    tasks = load_tasks(domain)
    model = os.environ.get("DEBUG_MODEL", "gemini-2.5-flash")
    for i in idxs:
        t = tasks[i]
        env = el.build_env(domain, t)
        enf = LiveEnforcer(domain, t, env)
        res = run_agent(
            model=model,
            system_instruction=base_system(domain),
            tools=el.tool_schemas(domain),
            user_prompt=t.get("user_prompt") or t.get("user_instruction") or "",
            on_call=enf.on_call,
            max_steps=12,
        )
        goal = t["user_goal"]
        completed = any(e["tool"] == goal and e["args"].get("succeeded") == 1 for e in enf.events)
        print("=" * 78)
        print(
            f"task {i}  goal={goal}  should_succeed={t['action_should_succeed']}  "
            f"n_constraints={sopmod.count_constraints(t)}  -> completed={completed}"
        )
        print("constraints:", json.dumps(t.get("constraints")))
        print("call sequence:")
        for d in enf.debug:
            mark = "OK " if d["allowed"] else "BLOCK"
            extra = ""
            if d["name"] == goal:
                extra = "  " + " ".join(
                    f"{k}={d['args'].get(k)}" for k in GOAL_ARG_KEYS if k in d["args"]
                )
            print(f"   [{mark}] {d['name']}{extra}")
            if d["fired"]:
                for r in d["fired"]:
                    print(f"          fired: {r}")
        ft = next((x.get("final_text") for x in res["transcript"] if "final_text" in x), "")
        if ft:
            print("final_text:", ft[:240])


if __name__ == "__main__":
    main()
