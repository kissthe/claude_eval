#!/usr/bin/env bash
# Mem0 on the 10-question subset with ONE model doing both memory construction and reading.
#   AIHUBMIX_API_KEY=... ./run_mem0_10.sh RUN_ID MODEL [reader flags]
# Reasoning models: export MEMEVAL_MEM0_LLM_EXTRA='{"is_reasoning_model": true}' and pass --reasoning.
set -euo pipefail
RUN=${1:?run id}; MODEL=${2:?model}; shift 2
cd "$(dirname "$0")"
: "${AIHUBMIX_API_KEY:?set AIHUBMIX_API_KEY}"
export MEMEVAL_LLM="$MODEL"
export MEMEVAL_SAMPLES=pilot_021,pilot_019,pilot_022,pilot_001,pilot_008,pilot_005,pilot_030,pilot_020,pilot_007,pilot_015
mkdir -p "runs/$RUN/logs"
pids=()
for c in conv-26 conv-30 conv-50; do
  .venv/bin/python run.py ingest --run "$RUN" --system mem0 --conv "$c" > "runs/$RUN/logs/ingest_mem0_$c.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
.venv/bin/python run.py contexts --run "$RUN" --conditions mem0 > "runs/$RUN/logs/contexts.log" 2>&1
.venv/bin/python run.py read --run "$RUN" --conditions mem0 --workers 10 "$@" > "runs/$RUN/logs/read.log" 2>&1
.venv/bin/python run.py score --run "$RUN" 2>/dev/null | grep -v "^Loaded\|Initializ"
