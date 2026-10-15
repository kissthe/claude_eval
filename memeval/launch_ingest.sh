#!/usr/bin/env bash
# Start all ingestion jobs (3 systems x 6 Pilot conversations) in parallel.
# The API key is read from the environment only (AIHUBMIX_API_KEY); it is never written to disk.
set -euo pipefail
RUN=${1:?usage: launch_ingest.sh RUN_ID [systems]}
SYSTEMS=${2:-"amem memoryos mem0"}
cd "$(dirname "$0")"
: "${AIHUBMIX_API_KEY:?set AIHUBMIX_API_KEY}"
mkdir -p "runs/$RUN/logs"
for s in $SYSTEMS; do
  for c in conv-26 conv-30 conv-43 conv-44 conv-48 conv-50; do
    nohup .venv/bin/python run.py ingest --run "$RUN" --system "$s" --conv "$c" \
      > "runs/$RUN/logs/ingest_${s}_${c}.log" 2>&1 &
  done
done
echo "launched; progress: tail -f runs/$RUN/logs/*.log"
