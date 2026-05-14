#!/usr/bin/env bash
# run_inference.sh — one-shot wrapper for scripts/run_inference.py
#
# Usage:
#   ./run_inference.sh my_voice                       # reads inference_dataset/inf_dataset_1/
#   ./run_inference.sh my_voice inf_dataset_2         # reads inference_dataset/inf_dataset_2/
#   ./run_inference.sh my_voice inf_dataset_1 --transpose 2
#   ./run_inference.sh my_voice --input path/to/file.wav    # override with explicit path
#
# All extra flags are forwarded verbatim to run_inference.py.

set -euo pipefail

MODEL_NAME="${1:-${MODEL_NAME:-my_voice}}"
if [[ $# -ge 1 && "$1" != --* ]]; then shift; fi

INFERENCE_DATASET="${INFERENCE_DATASET:-inf_dataset_1}"
if [[ $# -ge 1 && "$1" != --* ]]; then
    INFERENCE_DATASET="$1"
    shift
fi

TRANSPOSE="${TRANSPOSE:-0}"
F0METHOD="${F0METHOD:-rmvpe}"
INDEX_RATE="${INDEX_RATE:-0.66}"
DEVICE="${DEVICE:-cuda:0}"
IS_HALF="${IS_HALF:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/scripts/run_inference.py" \
    --model_name        "$MODEL_NAME" \
    --inference_dataset "$INFERENCE_DATASET" \
    --transpose         "$TRANSPOSE" \
    --f0method          "$F0METHOD" \
    --index_rate        "$INDEX_RATE" \
    --device            "$DEVICE" \
    --is_half           "$IS_HALF" \
    "$@"
