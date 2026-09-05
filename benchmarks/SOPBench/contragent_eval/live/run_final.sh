#!/bin/bash
# Final supplement run, rate-limit-aware:
#   PHASE 1: gemini-2.5-pro sweep (separate per-model quota) -- model-agnostic check.
#   PHASE 2: flash seed-3 remaining domains (after pro gives flash RPM time to cool).
# Low concurrency (workers=3) + generous-but-CAPPED retries so transient RPM
# throttles are ridden out without the old multi-hour hang. Per-domain 35-min
# watchdog as a backstop.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)" && set -a && source .env && set +a
cd benchmarks/SOPBench
PY=/opt/homebrew/opt/python@3.13/bin/python3.13
RES=contragent_eval/live/results
export GEMINI_HTTP_TIMEOUT=90 GEMINI_MAX_RETRIES=8 GEMINI_MAX_BACKOFF=45
TIMEOUT=2100

run_dom () {  # model pref limit think domain
  local model="$1" pref="$2" lim="$3" think="$4" d="$5"
  [ -f ${RES}/${pref}_${d}.json ] && { echo "  skip ${pref}_${d} (exists)"; return; }
  echo "=========== $pref $d ($(date +%H:%M:%S)) ==========="
  if [ "$think" = "default" ]; then unset GEMINI_THINKING; else export GEMINI_THINKING=0; fi
  PYTHONPATH=contragent_eval:.:../.. $PY contragent_eval/live/run.py \
    --domain "$d" --condition base prompt enforce llm_guard \
    --limit "$lim" --model "$model" --max-steps 12 --workers 3 \
    --out ${RES}/${pref}_${d}.json > ${RES}/${pref}_${d}.log 2>&1 &
  local pid=$!
  ( sleep $TIMEOUT; kill -9 $pid 2>/dev/null ) & local killer=$!
  wait $pid 2>/dev/null; local rc=$?
  kill $killer 2>/dev/null; wait $killer 2>/dev/null
  local n=$(grep -cE '^\[(OK|ERR)\]' ${RES}/${pref}_${d}.log 2>/dev/null)
  local e=$(grep -c '^\[ERR\]' ${RES}/${pref}_${d}.log 2>/dev/null)
  echo "  -> $pref $d done rc=$rc tasks=$n err=$e ($(date +%H:%M:%S))"
}

echo "######## PHASE 1: GEMINI-2.5-PRO ########"
for d in bank dmv healthcare hotel library online_market university; do
  run_dom gemini-2.5-pro pro 30 default "$d"
done

echo "######## PHASE 2: FLASH SEED 3 (remaining) ########"
for d in dmv healthcare hotel library online_market university; do
  run_dom gemini-2.5-flash s3 40 0 "$d"
done

echo "=========== FINAL DONE ($(date +%H:%M:%S)) ==========="
