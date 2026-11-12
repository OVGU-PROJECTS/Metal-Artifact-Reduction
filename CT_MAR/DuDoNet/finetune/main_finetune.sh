#!/usr/bin/env bash

DUDONET_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DUDONET_DIR" || exit 1

function release_on_exit() {
    # Only release if gpurequest returned a GPU
    if [[ -n "$ID" ]]; then
        gpurelease $ID
        gpurelease $I
        #gpurelease $S
        #gpurelease $A
    fi
}
trap release_on_exit EXIT

ID=$(gpurequest)
I=$(gpurequest)
#S=$(gpurequest)
#A=$(gpurequest)

if [[ -z "$ID" ]]; then
    echo "No GPU available. Exiting..."
    exit 1
fi

echo "allocated GPUs: $ID and $I"

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#debug sync
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1


# Split trials across 2 workers (total 50)
TRIALS_TOTAL=50
TRIALS_PER_WORKER=$(( (TRIALS_TOTAL + 1) / 2 ))   # 25

OUT_DIR="/mnt/public/bahaa/DuDoNet/smbo"
mkdir -p "$OUT_DIR/logs"

# ---------------- Worker 0 on GPU $ID ----------------
(
  export CUDA_VISIBLE_DEVICES=$ID
  unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

  # sanity: show what this worker sees
  python - <<'PY'
import os, torch
print("[worker0] CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[worker0] torch.cuda.device_count()=", torch.cuda.device_count())
print("[worker0] current_device=", torch.cuda.current_device())
PY

  python -m finetune.finetune \
    --storage sqlite:////mnt/public/bahaa/DuDoNet/smbo/dudonet_smbo.db \
    --study dudonet_phase1 \
    --train_root /mnt/public/bahaa/dataset_training \
    --val_root   /mnt/public/bahaa/dataset_validation \
    --out_dir    /mnt/public/bahaa/DuDoNet/smbo \
    --trials $TRIALS_PER_WORKER \
    --smbo_epochs 12 \
    --trainset 1000 \
    --valset 256 \
    --seed 42 \
    > "$OUT_DIR/logs/worker0_gpu${ID}.log" 2>&1
) &
P0=$!

# ---------------- Worker 1 on GPU $I ----------------
(
  export CUDA_VISIBLE_DEVICES=$I
  unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

  python - <<'PY'
import os, torch
print("[worker1] CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[worker1] torch.cuda.device_count()=", torch.cuda.device_count())
print("[worker1] current_device=", torch.cuda.current_device())
PY

  python -m finetune.finetune \
    --storage sqlite:////mnt/public/bahaa/DuDoNet/smbo/dudonet_smbo.db \
    --study dudonet_phase1 \
    --train_root /mnt/public/bahaa/dataset_training \
    --val_root   /mnt/public/bahaa/dataset_validation \
    --out_dir    /mnt/public/bahaa/DuDoNet/smbo \
    --trials $TRIALS_PER_WORKER \
    --smbo_epochs 12 \
    --trainset 1000 \
    --valset 256 \
    --seed 42 \
    > "$OUT_DIR/logs/worker1_gpu${I}.log" 2>&1
) &
P1=$!

echo "workers started: P0=$P0 P1=$P1"

wait $P0; EC0=$?
wait $P1; EC1=$?

echo "worker exit codes: worker0=$EC0 worker1=$EC1"
if [[ $EC0 -ne 0 ]]; then tail -n 80 "$OUT_DIR/logs/worker0_gpu${ID}.log"; fi
if [[ $EC1 -ne 0 ]]; then tail -n 80 "$OUT_DIR/logs/worker1_gpu${I}.log"; fi

# Final pass: write ranking.json once (no new trials)
(
  export CUDA_VISIBLE_DEVICES=$ID
  unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
  python -m finetune.finetune \
    --storage sqlite:////mnt/public/bahaa/DuDoNet/smbo/dudonet_smbo.db \
    --study dudonet_phase1 \
    --train_root /mnt/public/bahaa/dataset_training \
    --val_root   /mnt/public/bahaa/dataset_validation \
    --out_dir    /mnt/public/bahaa/DuDoNet/smbo \
    --trials 0 \
    --smbo_epochs 12 \
    --trainset 1000 \
    --valset 256 \
    --seed 42 \
    > "$OUT_DIR/logs/ranking_write.log" 2>&1
)


echo "done. check:"
echo "  $OUT_DIR/ranking.json"
echo "  $OUT_DIR/logs/"

