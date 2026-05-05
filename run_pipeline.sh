#!/usr/bin/env bash
# run_pipeline.sh — one-shot wrapper for scripts/run_pipeline.py
#
# Usage:
#   ./run_pipeline.sh my_voice
#   ./run_pipeline.sh my_voice --epochs 300 --sr 48k
#   MODEL_NAME=my_voice EPOCHS=300 ./run_pipeline.sh
#
# All flags are passed through to run_pipeline.py unchanged.
# Override defaults here or via env vars.

set -euo pipefail

MODEL_NAME="${1:-${MODEL_NAME:-my_voice}}"
SR="${SR:-40k}"
EPOCHS="${EPOCHS:-200}"
SAVE_EPOCH="${SAVE_EPOCH:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPUS="${GPUS:-0}"
F0METHOD="${F0METHOD:-rmvpe}"
N_PROCESSES="${N_PROCESSES:-4}"

# Shift past model name so extra flags can be forwarded verbatim
if [[ $# -ge 1 && "$1" != --* ]]; then
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/scripts/run_pipeline.py" \
    --model_name  "$MODEL_NAME" \
    --sr          "$SR" \
    --epochs      "$EPOCHS" \
    --save_epoch  "$SAVE_EPOCH" \
    --batch_size  "$BATCH_SIZE" \
    --gpus        "$GPUS" \
    --f0method    "$F0METHOD" \
    --n_processes "$N_PROCESSES" \
    "$@"
