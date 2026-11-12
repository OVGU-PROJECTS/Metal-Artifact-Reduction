# PConv-MAR

This folder contains the PyTorch partial-convolution (Pconv) sinogram MAR model.

## Structure

- `model/`: PConv layers, model definition, and combined loss.
- `data/`: dataset loader code.
- `training/`: config loading, DDP helpers, training loop, evaluation, logging, checkpointing.
- `data_inconsistency/`: Helgeson-ludwig data-consistency loss.
- `recon/`: reconstruction helpers.
- `tools/`: loss terms calibration and gradient strength analysis of each loss term.
- `experiments/`: the loss-weight sweep and ablation study.
- `finetune/`: Optuna hyperparameter search, run one stage at a time.
- `model/nvidia_partialconv2d.py`: NVIDIA's original published implementation,
  vendored unchanged from commit `a99cd7c` under its BSD license.
- `model/pconv_nvidia_layer.py`: active thin PConv-MAR interface/padding adapter
  around the vendored NVIDIA class, it contains no operator reimplementation.

## Setup

Create the shared CT-MAR environment from the repository root:

```bash
cd CT_MAR
conda env create -f configs/environment.yml
conda activate ct-mar-env
python -m pip install -e .
```

This environment can run PConv-MAR, DuDoNet, and CNN-MAR. It includes PyTorch/torchvision, CUDA, CuPy/Numba reconstruction helpers, `SimpleITK`, `gecatsim`, and other packages

## Config
Training uses:

```text
configs/pconv.yaml
```
Before running, edit:

- `train_root`
- `val_root`
- `test_root`
- `run_dir`
- `stats_path`
- `init_weights`
- `epochs`
- `batch_size`
- `n_gpus`
- loss weights and calibration values

## Training
Single-process training:

```bash
cd CT_MAR/Pconv_mar
python -m training.trainer --config pconv
```

Multi-GPU training:

```bash
cd CT_MAR/Pconv_mar
torchrun --standalone --nproc_per_node=4 -m training.trainer --config pconv
```

The provided `train.sh` is a cluster launcher. It reserves specific GPUs and uses cluster-specific commands, so inspect and edit it before running locally.

Training writes:

- `config.json`
- `checkpoints/`
- `csv/`
- `logs/`
- `tb/`

Training reports the calibrated loss terms (`rmse`, `hole`, `dc`, `img`, `edge`,
`perc`) and the thesis development monitors: `ssim_aapm`, `ssim_trace`,
`mae_near_metal`, and `streak_edge_err`. The loss terms and fine-tuning selection
score are unchanged by metric cleanup. See [METRICS.md](../METRICS.md).

Resuming checks CSV headers before appending. For historical logs with a different
metric schema, retain the original directory, copy its `checkpoints/last.pt` to
`checkpoints/last.pt` inside a new run directory, and set `run_dir` and
`initial_epoch` accordingly. The existing checkpoint loader restores model,
optimizer, and scheduler state from that file. New CSVs receive the current
headers.
