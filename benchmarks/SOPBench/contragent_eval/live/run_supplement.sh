#!/bin/bash
# Supplementary runs to harden the SOPBench live result:
#   (1) two more flash seeds (seed1 = final_<d>.json already exists) -> CI over 3 seeds
#   (2) a second, higher-capability base model (gemini-2.5-pro) -> model-agnostic check
# Same 4 conditions, same balanced slice. flash uses thinking=0 (matches seed1);
# pro uses DEFAULT thinking (pro rejects thinkingBudget=0).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)" && set -a && source .env && set +a
cd benchmarks/SOPBench
PY=/opt/homebrew/opt/python@3.13/bin/python3.13
DOMAINS="bank dmv healthcare hotel library online_market university"
RES=contragent_eval/live/results

run_sweep () {  # $1=model $2=outprefix $3=limit $4=thinking(0|default)
  local model="$1" pref="$2" lim="$3" think="$4"
  for d in $DOMAINS; do
    echo "=========== $pref $d ($(date +%H:%M:%S)) ==========="
    if [ "$think" = "default" ]; then unset GEMINI_THINKING; else export GEMINI_THINKING=0; fi
    PYTHONPATH=contragent_eval:.:../.. $PY contragent_eval/live/run.py \
      --domain "$d" --condition base prompt enforce llm_guard \
      --limit "$lim" --model "$model" --max-steps 12 --workers 6 \
      --out ${RES}/${pref}_${d}.json 2>&1 | tail -16
  done
}

echo "######## FLASH SEED 2 ########"
run_sweep gemini-2.5-flash s2 40 0
echo "######## FLASH SEED 3 ########"
run_sweep gemini-2.5-flash s3 40 0
echo "######## GEMINI-2.5-PRO (model-agnostic) ########"
run_sweep gemini-2.5-pro pro 30 default
echo "=========== SUPPLEMENT DONE ($(date +%H:%M:%S)) ==========="
