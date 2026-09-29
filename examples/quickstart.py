"""The README quickstart, runnable: load a library, gate three calls.

    python3 examples/quickstart.py

No model and no credentials: the calls are made by hand so the output
shows exactly what the supervisor decides at each one.
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from contragent import ContrAgent  # noqa: E402

guard = ContrAgent(agent_id="bank_agent", config=_ROOT / "examples" / "bank.yaml")

# 1. A transfer before any verification: refused, and the feedback names
#    the rule in the words of the library, ready to hand back to the model.
result = guard.guard_before("transfer_funds", {"amount": 500})
print("transfer_funds blocked:", result.blocked)
print("  feedback:", result.feedback)

# 2. Verify first. guard_before admits the call; guard_after records its result.
guard.guard_before("verify_identity", {})
guard.guard_after("verify_identity", {"ok": True})

# 3. Now the transfer passes.
result = guard.guard_before("transfer_funds", {"amount": 500})
print("transfer_funds blocked:", result.blocked)
guard.guard_after("transfer_funds", {"ok": True})

# 4. End of session. The receipt was never sent, so the eventuality
#    "every transfer is eventually receipted" is reported as violated.
for verdict in guard.finish_session():
    print("pending at session end:", verdict.desc)
