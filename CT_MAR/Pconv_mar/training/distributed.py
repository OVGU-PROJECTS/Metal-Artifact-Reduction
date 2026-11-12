"""Device and DDP helpers for training/trainer.py."""
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen=True)
class DistContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    is_main: bool


def to_device(batch, device):
    (proj, mask), y_true = batch
    return ((proj.to(device, non_blocking=True), mask.to(device, non_blocking=True)),
            y_true.to(device, non_blocking=True))


def unwrap_model(model):
    """the bare module behind a DDP wrapper"""
    return model.module if isinstance(model, DDP) else model


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def reduce_mean(value_sum, value_count, device):
    """average a sum/count pair across all DDP ranks
    (so logged metrics are global)
    """
    stats = torch.tensor([value_sum, value_count], device=device, dtype=torch.float64)
    if is_distributed():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[1]).item()


def setup_distributed(n_gpus):
    """ 
    - read torchrun's env vars and init the process group
    - returns a DistContext, and warns if n_gpus doesn't match the launch
    - Falls back to a single-process (1 GPU) context when not launched via torchrun.
    - CUDA is required
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training but no GPU is available.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")

    ctx = DistContext(distributed=distributed, rank=rank, world_size=world_size,
                      local_rank=local_rank, device=torch.device("cuda", local_rank),
                      is_main=(rank == 0))
    if ctx.is_main and n_gpus != world_size:
        print(f"[train] configs n_gpus={n_gpus} but WORLD_SIZE={world_size}. "
              "launch with torchrun --nproc_per_node=<gpu_count> for true multi-GPU DDP.",
              flush=True)
    return ctx
