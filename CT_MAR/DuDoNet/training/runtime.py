"""Run setup helpers for DuDoNet training """
import datetime
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import yaml


def _load_cfg(name):
    """Load configs/<name>.yaml."""
    root = Path(os.environ.get(
        "MAR_CONFIG_DIR",
        Path(__file__).resolve().parents[2] / "configs",
    ))
    data = yaml.safe_load((root / f"{name}.yaml").read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{name}.yaml must contain a mapping")
    return SimpleNamespace(**data)


def ddp_setup():
    """Initialize DDP when LOCAL_RANK is set."""
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 0, 1  # (is_ddp, local_rank, rank, world_size)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(minutes=60),
    )
    return True, local_rank, rank, world_size


def ddp_cleanup(is_ddp: bool):
    if is_ddp:
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model
