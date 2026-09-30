"""The README quickstart, runnable: load a library, gate a few calls.

    python3 examples/quickstart.py

No model and no credentials: the calls are made by hand, and
``verbose=True`` prints the session as a timeline as it happens.
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from contragent import ContrAgent  # noqa: E402

guard = ContrAgent(agent_id="bank_agent", config=_ROOT / "examples" / "bank.yaml", verbose=True)

# A transfer before any verification is refused; the feedback names the rule.
guard.guard_before("transfer_funds", {"amount": 500, "to": "ACME"})

# Verify first: guard_before admits the call, guard_after records its result.
guard.guard_before("verify_identity", {"user_id": "u1"})
guard.guard_after("verify_identity", {"ok": True})

# Now the transfer passes.
guard.guard_before("transfer_funds", {"amount": 500, "to": "ACME"})
guard.guard_after("transfer_funds", {"ok": True})

# A file read whose result carries a private key: the call ran, the result
# is withheld from the model (the assumption is kept for it).
guard.guard_before("read_file", {"path": "id_rsa"})
guard.guard_after("read_file", "-----BEGIN PRIVATE KEY-----")

# After a file read, no email.
guard.guard_before("send_email", {"to": "x", "body": "hi"})

# End of session: the receipt was never sent, so that obligation is reported.
guard.finish_session()
