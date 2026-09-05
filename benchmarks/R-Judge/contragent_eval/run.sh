#!/usr/bin/env bash
# Convert R-Judge records to ContrAgent traces, then evaluate the hand-authored
# contracts and print the confusion matrix.
#
# Usage:  bash run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

echo "== Converting R-Judge data -> traces/ =="
python3 "$HERE/convert.py"

echo
echo "== Evaluating contracts (overall confusion matrix) =="
cd "$REPO"
PYTHONPATH=. python3 -m contragent.cli eval \
  "benchmarks/R-Judge/contragent_eval/traces" \
  --config "benchmarks/R-Judge/contragent_eval/contracts.yaml" \
  --agent "*"

echo
echo "== Per-category recall / FPR =="
PYTHONPATH=. python3 "$HERE/score_by_category.py"
