"""Output artifacts helpers for DuDoNet  training"""
import csv
import json
from pathlib import Path

import torch

from .runtime import unwrap_model


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def csv_init(path: Path, header: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header != header:
            raise ValueError(
                f"Cannot append to {path}: metric columns differ. "
                "Preserve the historical log and use a new output directory."
            )
    else:
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)


def csv_append(path: Path, row: list):
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)


def save_ckpt(path: Path, model, opt, sched, epoch: int, best_val: float, extra: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model_state": unwrap_model(model).state_dict(),
        "opt_state": opt.state_dict(),
        "sched_state": sched.state_dict() if sched is not None else None,
    }
    if extra is not None:
        state["extra"] = extra
    torch.save(state, str(path))
