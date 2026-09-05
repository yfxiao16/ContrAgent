#!/usr/bin/env bash
# Run AgentPex import + eval for one domain's sample file.
# Usage: run_domain.sh <domain> <run_tag>
#   domain  : retail | airline | telecom
#   run_tag : label for this run (e.g. main, var1, var2) -> result subdir
set -euo pipefail

DOMAIN="$1"
TAG="${2:-main}"
CMP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="/tmp/agentpex/src"
PARALLEL="${PARALLEL:-6}"

source /tmp/agentpex/venv/bin/activate
cd "$SRC"
cp "$CMP_DIR/${DOMAIN}_sample.json" "./${DOMAIN}_${TAG}.json"

echo "[$(date +%T)] IMPORT $DOMAIN/$TAG"
IMPORT_OUT=$(python import_traces.py "${DOMAIN}_${TAG}.json" --tau-sq 2>&1)
ART=$(echo "$IMPORT_OUT" | grep "Artifact directory:" | tail -1 | sed 's/.*Artifact directory: //' | tr -d '[:space:]')
echo "[$(date +%T)] artifact: $ART"
[ -n "$ART" ] || { echo "IMPORT FAILED"; echo "$IMPORT_OUT" | tail -20; exit 1; }

echo "[$(date +%T)] EVAL $DOMAIN/$TAG (parallel=$PARALLEL)"
python cli.py --skip-sample --gen-eval --input-test-result "$ART" \
    --num-parallel-evals "$PARALLEL" 2>&1 | tail -5

# stash the eval artifact dir path for scoring
echo "$ART" > "$CMP_DIR/artifact_${DOMAIN}_${TAG}.path"
echo "[$(date +%T)] DONE $DOMAIN/$TAG -> $ART"
