#!/bin/bash
# Clean, internally-consistent 4-condition SOPBench sweep on ONE model.
# base / prompt / enforce(+latch) / llm_guard, same model + same balanced slice,
# so the whole tab:sopbench-main is comparable and the llm-guard column +
# latch-on enforce land together. Writes live/results/final_<domain>.json.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)" && set -a && source .env && set +a
cd benchmarks/SOPBench
export GEMINI_THINKING=0          # flash is a capable function-caller w/o thinking
export GEMINI_CALL_DELAY=0
MODEL=gemini-2.5-flash
LIMIT=40                          # balanced 20 pos / 20 neg (small domains use all)
PY=python3

for d in bank dmv healthcare hotel library online_market university; do
  echo "=========== $d ($(date +%H:%M:%S)) ==========="
  PYTHONPATH=contragent_eval:.:../.. $PY contragent_eval/live/run.py \
    --domain "$d" --condition base prompt enforce llm_guard \
    --limit $LIMIT --model $MODEL --max-steps 12 --workers 6 \
    --out contragent_eval/live/results/final_${d}.json \
    2>&1 | tail -40
done
echo "=========== SWEEP DONE ($(date +%H:%M:%S)) ==========="
