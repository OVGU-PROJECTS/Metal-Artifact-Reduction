# DuDoNet

This folder is a PyTorch reimplementation of DuDoNet, built from the paper:
Lin et al., "DuDoNet: Dual Domain Network for CT Metal Artifact Reduction", CVPR 2019,
https://doi.org/10.1109/CVPR.2019.01076

## Structure
- `models/`: DuDoNet model definition and loss.
- `modules/`: image-domain and sinogram-domain network modules.
- `training/`: training loop, DDP setup, evaluation, and artifact writing.
- `finetune/`: Optuna learning-rate search scripts.
- `evaluation/`: test-set inference and clinical submission generation.

## Setup

Create the shared CT-MAR environment from the repository root:

```bash
cd CT_MAR
conda env create -f configs/environment.yml
conda activate ct-mar-env
python -m pip install -e .
```

This environment can run DuDoNet, PConv-MAR, and cnn_mar. It includes PyTorch, CUDA, the CuPy/Numba reconstruction layer, and other required packages.
`python -m pip install -e .` registers the local shared packages from `src/`.

## Config

Training uses:
```text
configs/dudonet.yaml
```

Before running, edit:

- `train_root`
- `val_root`
- `out_dir`
- `epochs`
- `batch_size`
- `lr`
- `trainset` and `valset` if you want a subset

## Finetuning

The thesis reports one learning-rate search: 50 trials of 12 epochs each, with
1,000 training cases, 256 validation cases, fixed Adam betas `(0.5, 0.999)`, and
learning rates sampled over `[5e-5, 1e-3]`. The score is the lowest validation total
loss. Run from this model directory:

```bash
cd CT_MAR/DuDoNet
python -m finetune.finetune \
  --train_root /path/to/training --val_root /path/to/validation \
  --storage sqlite:////path/to/new-study.db \
  --study dudonet_phase1 --out_dir /path/to/new-search \
  --trials 50 --smbo_epochs 12 --trainset 1000 --valset 256 --seed 42
```

`--trials` adds that many trials to the selected study, use a fresh database/study
for a new 50-trial search. With `--trials 0`, an existing completed study is exported
to `ranking.json` without training. CLI help and ranking export do not import the
CUDA trainer.

`finetune/main_finetune.sh` is the two-worker cluster launcher. It runs the same
first-stage search, then exports its ranking. Its GPU reservation commands and
absolute data/storage paths need adapting to your machine. The unreported
top-five robustness stage and its dedicated launcher have been removed.

Historical `smbo/phase1/` results, the two worker logs and `smbo/dudonet_smbo.db`
(the `dudonet_phase1` study only) are preserved. Use the pinned Optuna environment
for that database, or upgrade a copy when testing with newer Optuna versions.

## Training

Single GPU:
```bash
cd CT_MAR/DuDoNet
python -m training.trainer --config dudonet
```

Multi-GPU:

```bash
cd CT_MAR/DuDoNet
torchrun --standalone --nproc_per_node=4 -m training.trainer --config dudonet
```

Training writes:

- `config.json`
- `metrics.csv`
- `checkpoints/last.pt`
- `checkpoints/best.pt`
- periodic `epoch_*.pt` checkpoints

Training reports the original loss components and train/validation AAPM SSIM.
Existing CSVs with different columns are rejected before appending, use a new
output directory for the current schema. See [METRICS.md](../METRICS.md).

