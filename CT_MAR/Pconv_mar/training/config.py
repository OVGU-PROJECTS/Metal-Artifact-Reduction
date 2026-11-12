"""run-config loading + validation for training/trainer.py"""
import os
import yaml
from pathlib import Path
from types import SimpleNamespace

# Required trainer keys, kept in sync with the maintained PConv configs.
trainer_keys = {
    "run_dir", "stats_path", "init_weights",
    "train_root", "val_root", "test_root",
    "epochs", "initial_epoch", "batch_size", "n_train", "n_val", "seed",
    "num_workers", "lr", "decay_steps", "n_gpus", "grad_accum_steps",
    "rmse_scale", "hole_scale", "dc_scale", "img_scale",
    "edge_scale", "perc_scale",
    "start_ep_hole", "start_ep_image", "start_ep_edge", "start_ep_dc",
    "start_ep_perc", "ramp_duration", "ramp_duration_dc",
    "w_rmse", "w_hole", "w_image", "w_edge", "w_dc", "w_perc", "dc_mode",
    "bd_every_epochs", "bd_val_steps", "bd_train_steps",
    "log_every_steps", "save_every_epochs",
}


def check_keys(data, label):
    """a typo'd or missing key must fail loudly, not silently use a stale value"""
    extra, missing = set(data) - trainer_keys, trainer_keys - set(data)
    if extra or missing:
        raise ValueError(f"{label} key mismatch: extra={sorted(extra)} missing={sorted(missing)}")


def load_config(name):
    """Load + validate configs/<name>.yaml (or $MAR_CONFIG_DIR) as a namespace."""
    root = Path(os.environ.get(
        "MAR_CONFIG_DIR",
        Path(__file__).resolve().parents[2] / "configs",
    ))
    data = yaml.safe_load((root / f"{name}.yaml").read_text())
    check_keys(data, f"{name}.yaml")

    args = SimpleNamespace(**data)
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    return args
