set -e
# Run from CT_MAR/Pconv_mar. The phases are phase1 -> dc -> phase2.

PHASE="${1:?Use phase1, dc, or phase2}"
STUDY_ROOT="${STUDY_ROOT:-experiments/results/phase1}"
export MAR_CONFIG_DIR="${MAR_CONFIG_DIR:-../configs/loss_study_configs}"

ANCHOR_EPOCHS=5
RUN_EPOCHS=15
WARMUP_STEPS=500
CALIB_STEPS=1000
GRAD_STEPS=50
SEED=42
LOSS_MULTS="${LOSS_MULTS:-0.25,0.5,1,2}"
DC_MULTS="${DC_MULTS:-0.0025,0.005,0.01,0.02}"
DC_MODES=(
    direct_m0
    direct_m0_m1
    direct_m0_m1_m2_light
)
N_GPUS=4

# run as a module so it can import the shared training.terms table
GEN=experiments.loss_study.gen_configs
CONFIG="$MAR_CONFIG_DIR/pconv.yaml"
ANCHOR="$STUDY_ROOT/anchor/checkpoints/best.pt"
DC_ROOT="$STUDY_ROOT/dc_modes"

case "$PHASE" in
    phase1) ;;
    dc)
        if [ ! -f "$ANCHOR" ]; then
            echo "Missing shared anchor checkpoint: $ANCHOR" >&2
            echo "Run phase1 first." >&2
            exit 2
        fi
        ;;
    phase2)
        if [ "${PHASE2_CONFIG_READY:-0}" != 1 ]; then
            echo "Edit $CONFIG, then rerun with PHASE2_CONFIG_READY=1." >&2
            exit 2
        fi
        ;;
    *) echo "Use phase1, dc, or phase2." >&2; exit 2 ;;
esac

# Effective batch: 2 x 4 GPUs x 4 accumulation = 32.
perl -pi -e 's/^n_gpus: .*/n_gpus: 4/' "$CONFIG"
perl -pi -e 's/^grad_accum_steps: .*/grad_accum_steps: 4/' "$CONFIG"

export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"

train_config() {
    local config_name=$1
    local port
    port=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
    torchrun --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:$port" \
      --rdzv-id="loss_$$_${RANDOM}" --nproc_per_node="$N_GPUS" \
      -m training.trainer --config "$config_name"
}

train_list() {
    while IFS= read -r config_name; do
        [ -n "$config_name" ] && train_config "$config_name"
    done < "$1"
}

run_dc_studies() {
    local mode
    local root

    mkdir -p "$DC_ROOT"
    printf '%s\n' "${DC_MODES[@]}" > "$DC_ROOT/direct_modes.txt"

    for mode in "${DC_MODES[@]}"; do
        root="$DC_ROOT/$mode"
        mkdir -p "$root"

        python -m "$GEN" apply --dc_mode "$mode"
        python -m tools.calibrate calib --config pconv --split train \
          --warmup_steps "$WARMUP_STEPS" --steps "$CALIB_STEPS" \
          --output_json "$root/fixed_calib.json"
        python -m "$GEN" apply --calib_json "$root/fixed_calib.json" --calib_terms dc

        python -m tools.calibrate grad --config pconv --split train \
          --weights "$ANCHOR" --steps "$GRAD_STEPS" \
          --output_json "$root/anchor_grad.json"
        python -m "$GEN" apply --grad_json "$root/anchor_grad.json"

        python -m "$GEN" sweeps --study_root "$root" --seed "$SEED" \
          --anchor_ckpt "$ANCHOR" --branch_epochs "$RUN_EPOCHS" \
          --terms dc --mults "$DC_MULTS" --skip_b0
        train_list "$root/pilot_configs.txt"
    done
}

if [ "$PHASE" = phase1 ]; then
    mkdir -p "$STUDY_ROOT"

    # Calibrate, train the shared anchor, then sweep reconstruction losses.
    python -m tools.calibrate calib --config pconv --split train \
      --warmup_steps "$WARMUP_STEPS" --steps "$CALIB_STEPS" \
      --output_json "$STUDY_ROOT/fixed_calib.json"
    python -m "$GEN" apply --calib_json "$STUDY_ROOT/fixed_calib.json"

    python -m "$GEN" anchor --study_root "$STUDY_ROOT" --seed "$SEED" \
      --anchor_epochs "$ANCHOR_EPOCHS"
    train_config ls_anchor

    python -m tools.calibrate grad --config pconv --split train \
      --weights "$ANCHOR" --steps "$GRAD_STEPS" \
      --output_json "$STUDY_ROOT/anchor_grad.json"
    python -m "$GEN" apply --grad_json "$STUDY_ROOT/anchor_grad.json"

    python -m "$GEN" sweeps --study_root "$STUDY_ROOT" --seed "$SEED" \
      --anchor_ckpt "$ANCHOR" --branch_epochs "$RUN_EPOCHS" \
      --terms img,edge,perc --mults "$LOSS_MULTS"
    train_list "$STUDY_ROOT/pilot_configs.txt"

    echo "Phase 1 complete. Run the dc phase next."
elif [ "$PHASE" = dc ]; then
    run_dc_studies
    echo "DC studies complete. Stop and manually set in $CONFIG:"
    echo "w_image, w_edge, w_perc, dc_mode, dc_scale, w_dc"
else
    # Phase 2 uses the weights selected manually after the sweeps.
    grep -nE '^(dc_mode|dc_scale|w_image|w_edge|w_dc|w_perc):' "$CONFIG"
    python -m "$GEN" final_ablations --study_root "$STUDY_ROOT" --seed "$SEED" \
      --anchor_ckpt "$ANCHOR" --branch_epochs "$RUN_EPOCHS"
    train_list "$STUDY_ROOT/final_ablation_configs.txt"
fi
