#!/usr/bin/env bash
set -euo pipefail
# Run (or resume) one Optuna stage of the hyperparameter search. Run from CT_MAR/Pconv_mar.
#   bash finetune/run_optuna_stage.sh 01_projection_optimizer 20
# Stages are chained by hand: feed the previous stage's exported best_config.json as OPT_BASE.
#   OPT_BASE=finetune/results/01_projection_optimizer/best_config.json \
#       bash finetune/run_optuna_stage.sh 02_reconstruction_losses 20
# 03_dc_loss , 04_curriculum, 05_final_confirm
GPUS=4,5
STAGE="${1:?Use a stage name, e.g. 01_projection_optimizer}"
TRIALS="${2:-12}"
BASE="${OPT_BASE:-finetune/configs/base.json}"

export CUDA_VISIBLE_DEVICES="$GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"

# resuming needs no flag: the study database lives in --study-dir and is reopened if present,
# so re-running with a larger --trials just adds the missing trials.
python -m finetune.cli \
    --base "$BASE" \
    --stage "finetune/configs/stages/${STAGE}.json" \
    --study-dir "finetune/resultss/${STAGE}" \
    --trials "$TRIALS"
