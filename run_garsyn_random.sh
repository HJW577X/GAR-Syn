#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hjw/anaconda3/envs/synergyx/bin/python}"
DATA_DIR="${DATA_DIR:-./data_random}"
SOURCE_RAW_DIR="${SOURCE_RAW_DIR:-}"
RESULTS_DIR="${RESULTS_DIR:-./results_random}"
LOG_DIR="${LOG_DIR:-./logs_random}"
FOLDS="${FOLDS:-1,2,3,4,5}"
LABELS="${LABELS:-S_mean}"
GPU="${GPU:-1}"
GPU_CACHE="${GPU_CACHE:-all}"
GRAPH_CACHE="${GRAPH_CACHE:-full_batch}"
BATCH_SIZE="${BATCH_SIZE:-1024}"

cd "$ROOT_DIR"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

if [[ ! -d "$DATA_DIR/folds" ]]; then
  if [[ -z "$SOURCE_RAW_DIR" ]]; then
    echo "ERROR: SOURCE_RAW_DIR is required to prepare data." >&2
    echo "Example: SOURCE_RAW_DIR=/path/to/rawData/drugcomb bash run_garsyn_random.sh" >&2
    exit 1
  fi
  "$PYTHON_BIN" run.py prepare \
    --dataset drugcomb \
    --split-mode random \
    --source-raw-dir "$SOURCE_RAW_DIR" \
    --output-dir "$DATA_DIR" \
    --folds "$FOLDS" \
    2>&1 | tee "$LOG_DIR/prepare.log"
fi

"$PYTHON_BIN" run.py train \
  --config ./configs/garsyn_default.json \
  --labels "$LABELS" \
  --folds "$FOLDS" \
  --data-dir "$DATA_DIR" \
  --results-dir "$RESULTS_DIR" \
  --gpu "$GPU" \
  --gpu-cache "$GPU_CACHE" \
  --graph-cache-mode "$GRAPH_CACHE" \
  --batch-size "$BATCH_SIZE" \
  2>&1 | tee "$LOG_DIR/train.log"

"$PYTHON_BIN" run.py evaluate \
  --results-dir "$RESULTS_DIR" \
  --labels "$LABELS" \
  --output "$RESULTS_DIR/summary_all.csv" \
  2>&1 | tee "$LOG_DIR/evaluate.log"
