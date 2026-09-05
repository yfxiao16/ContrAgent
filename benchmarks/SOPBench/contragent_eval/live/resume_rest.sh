#!/usr/bin/env bash
# Resume the remaining SOPBench live domains once Gemini quota has recovered.
# Quota-safe: 2 workers + a few seconds between calls to stay under free-tier RPM.
# Run from anywhere:  bash benchmarks/SOPBench/contragent_eval/live/resume_rest.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)" && set -a && source .env && set +a
cd benchmarks/SOPBench
# RPM guard. Free-tier gemini-2.5-flash is ~10 RPM, so multi-step agent loops
# burst over it. After a daily-quota reset you can drop DELAY to 1-2 and raise
# --workers; while throttled keep DELAY high / workers low. Enable billing for
# a higher RPM tier to run fast.
export GEMINI_CALL_DELAY=3
for d in library university online_market; do
  echo "=== running $d ==="
  PYTHONPATH=contragent_eval:.:../.. /opt/homebrew/opt/python@3.13/bin/python3.13 \
    contragent_eval/live/run.py --domain "$d" --condition base prompt enforce \
    --workers 2 --model gemini-2.5-flash \
    --out contragent_eval/live/results/${d}.json
done
echo "REST DOMAINS DONE"
