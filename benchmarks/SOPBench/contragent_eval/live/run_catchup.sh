#!/bin/bash
# Catch-up after a hung domain: finish flash seed-3 (remaining domains) + the
# gemini-2.5-pro sweep. Each domain is wrapped in a 30-min watchdog so a single
# stuck task can't stall the whole run again (the domain is killed and skipped;
# partial files are simply re-runnable later).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)" && set -a && source .env && set +a
cd benchmarks/SOPBench
PY=/opt/homebrew/opt/python@3.13/bin/python3.13
RES=contragent_eval/live/results
TIMEOUT=1800

run_dom () {  # $1=model $2=pref $3=limit $4=think(0|default) $5=domain
  local model="$1" pref="$2" lim="$3" think="$4" d="$5"
  echo "=========== $pref $d ($(date +%H:%M:%S)) ==========="
  if [ "$think" = "default" ]; then unset GEMINI_THINKING; else export GEMINI_THINKING=0; fi
  PYTHONPATH=contragent_eval:.:../.. $PY contragent_eval/live/run.py \
    --domain "$d" --condition base prompt enforce llm_guard \
    --limit "$lim" --model "$model" --max-steps 12 --workers 6 \
    --out ${RES}/${pref}_${d}.json > ${RES}/${pref}_${d}.log 2>&1 &
  local pid=$!
  ( sleep $TIMEOUT; kill -9 $pid 2>/dev/null ) & local killer=$!
  wait $pid 2>/dev/null
  local rc=$?
  kill $killer 2>/dev/null; wait $killer 2>/dev/null
  if [ -f ${RES}/${pref}_${d}.json ]; then echo "  -> $pref $d OK ($(date +%H:%M:%S))"; else echo "  -> $pref $d FAILED/timeout rc=$rc"; fi
}

echo "######## FLASH SEED 3 (remaining) ########"
for d in dmv healthcare hotel library online_market university; do
  run_dom gemini-2.5-flash s3 40 0 "$d"
done

echo "######## GEMINI-2.5-PRO ########"
for d in bank dmv healthcare hotel library online_market university; do
  run_dom gemini-2.5-pro pro 30 default "$d"
done

echo "=========== CATCHUP DONE ($(date +%H:%M:%S)) ==========="
