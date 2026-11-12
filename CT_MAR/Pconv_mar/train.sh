#!/usr/bin/env bash
set -euo pipefail
# Run from CT_MAR/Pconv_mar.

GPUS=(2 3 4 5)
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"

CONFIG="${CONFIG:-pconv_hole_img}"
N_GPUS=${#GPUS[@]}
MAR_CONFIG_DIR="${MAR_CONFIG_DIR:-../configs}"
export MAR_CONFIG_DIR

if [ ! -f "$MAR_CONFIG_DIR/$CONFIG.yaml" ]; then
    echo "Missing training config: $MAR_CONFIG_DIR/$CONFIG.yaml" >&2
    exit 2
fi

torchrun \
  --standalone \
  --nproc_per_node="$N_GPUS" \
  -m training.trainer \
  --config "$CONFIG"
