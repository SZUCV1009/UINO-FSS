#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ] || ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo 'Usage: bash train_distillation.sh GPU_NUM [GPU_IDS] [SAM_SIZE] [DINO_SIZE] [STOP_LAYER] [OUTPUT_DIR] [INITIAL_CHECKPOINT] [extra args...]' >&2
  exit 2
fi
GPU_NUM=$1
GPU_IDS=${2:-}
if [ -z "$GPU_IDS" ]; then
  for ((i=0; i<GPU_NUM; i++)); do GPU_IDS+="${GPU_IDS:+,}$i"; done
fi
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [ "${#GPUS[@]}" -ne "$GPU_NUM" ]; then
  echo 'GPU_NUM must match the number of GPU_IDS.' >&2
  exit 2
fi
SAM_SIZE=${3:-592}
DINO_SIZE=${4:-518}
STOP_LAYER=${5:-4}
OUTPUT_DIR=${6:-exp_loss}
INITIAL=${7:-}
EXTRA_ARGS=("${@:8}")
INITIAL_ARGS=()
if [ -n "$INITIAL" ]; then INITIAL_ARGS=(--load_distilled_model --distilled_model_path "$INITIAL"); fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OMP_NUM_THREADS=1
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$GPU_NUM" \
  train_distillation.py --simg_size "$SAM_SIZE" --dimg_size "$DINO_SIZE" \
  --stop_layer "$STOP_LAYER" --root_path "$OUTPUT_DIR" --ba_train_flag \
  "${INITIAL_ARGS[@]}" "${EXTRA_ARGS[@]}"
