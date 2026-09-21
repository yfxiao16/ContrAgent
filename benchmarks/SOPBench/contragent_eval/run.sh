#!/usr/bin/env bash
# SOPBench -> ContrAgent evaluation harness.
#
# For each domain: convert the recorded agent trajectories under
# output/<domain>/ into labelled ContrAgent traces (traces/<domain>/), then
# evaluate the hand-authored SOP contracts (contragent/contracts/sopbench/<domain>.yaml) and print
# the confusion matrix.
#
# Label convention (read by `contragent eval` from the filename prefix):
#   safe_*   the SOP permits the outcome the agent produced (or it correctly
#            refused / never completed the goal)   -> should pass every contract
#   unsafe_* the agent completed a goal the SOP forbids (precondition unmet)
#            -> should be blocked by >= 1 contract
#
# Usage:
#   bash run.sh                 # all domains
#   bash run.sh bank healthcare # selected domains
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

DOMAINS=("$@")
if [ ${#DOMAINS[@]} -eq 0 ]; then
  DOMAINS=(bank healthcare online_market library dmv hotel university)
fi

echo "== Converting SOPBench trajectories -> traces/ =="
python3 "$HERE/convert.py" "${DOMAINS[@]}"

cd "$REPO"
for d in "${DOMAINS[@]}"; do
  cfg="contragent/contracts/sopbench/${d}.yaml"
  traces="benchmarks/SOPBench/contragent_eval/traces/${d}"
  if [ ! -f "$cfg" ]; then
    echo; echo "== ${d}: no contragent/contracts/sopbench/${d}.yaml — skipping =="; continue
  fi
  echo; echo "================ ${d} ================"
  PYTHONPATH=. python3 -m contragent.cli eval "$traces" --config "$cfg" --agent "*" \
    2>/dev/null
done
