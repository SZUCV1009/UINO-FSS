#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo 'Usage: bash train.sh GPU_IDS [fold0] [coco|pascal] [LOGPATH] [DISTILLED_CHECKPOINT] [extra args...]' >&2
  exit 2
fi
GPU_IDS=$1
FOLD=${2:-fold0}
BENCHMARK=${3:-coco}
LOGPATH=${4:-./logs/train}
DISTILLED=${5:-./checkpoints/distilled_base.pth}
EXTRA_ARGS=("${@:6}")
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OMP_NUM_THREADS=1
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#GPUS[@]}" \
  train.py --fold "${FOLD#fold}" --benchmark "$BENCHMARK" --logpath "$LOGPATH" \
  --stop_layer 4 --distilled_model_path "$DISTILLED" "${EXTRA_ARGS[@]}"
