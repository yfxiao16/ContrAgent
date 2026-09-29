# Examples

All three run without a model or credentials. To write a library of your own, start with [`docs/authoring.md`](../docs/authoring.md).

| File | Shows |
|---|---|
| [`quickstart.py`](quickstart.py) | The README program: load [`bank.yaml`](bank.yaml), refuse a premature transfer, admit it after verification, report the receipt still owed at session end. |
| [`agent_loop.py`](agent_loop.py) | The integration shape: your own tool-calling loop with `guard_before` in front of each call and `guard_after` behind it, refusals fed back to the model, contracts written with the fluent helper. |
| [`traces/`](traces/) | Two recorded traces for the offline role: replay them against [`refund.yaml`](refund.yaml). |

```bash
python3 examples/quickstart.py
python3 examples/agent_loop.py
contragent replay examples/traces/refund_after_check.json  --config examples/refund.yaml   # PASS
contragent replay examples/traces/refund_without_check.json --config examples/refund.yaml  # FAIL
```

A trace is a JSON object with an `events` list, one entry per event in
order: `{"ts": 0, "agent": "agent", "type": "tool_call", "tool": "check_policy",
"args": {...}}`. The `agent` field must match `--agent` (default `agent`).
`contragent eval DIR --config ...` scores a whole directory of
`safe_*.json` / `unsafe_*.json` files the same way; the benchmark
harnesses under `benchmarks/` convert each dataset into this format.
