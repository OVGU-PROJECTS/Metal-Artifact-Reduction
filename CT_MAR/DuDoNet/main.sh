
function release_on_exit() {
    if [[ -n "$ID" ]]; then
        gpurelease $ID
        gpurelease $I
        gpurelease $S
        gpurelease $N
    fi
}
trap release_on_exit EXIT

ID=$(gpurequest)
I=$(gpurequest)
S=$(gpurequest)
N=$(gpurequest)

if [[ -z "$ID" || -z "$I" ]]; then
    echo "Need 2 GPUs. Got: ID=$ID I=$I"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$ID,$I,$S,$N
echo "using GPUs: $CUDA_VISIBLE_DEVICES"

# keep your cluster stability flags
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# good defaults
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1

# (optional but recommended for consistent numbering)
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# run DDP (2 processes, one per GPU)
torchrun --standalone --nproc_per_node=4 -m training.trainer
