"""launch one training subprocess for a trial."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

from .config import TRIAL_CONFIG_NAME

# Repo root, the parent of finetune. Trials run with cwd exsit here so
# "-m training.trainer" resolves its imports the usual way
REPO_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    """to avoid two runs crashing, grab a free TCP port"""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def build_command(n_gpus: int) -> list[str]:
    if n_gpus > 1:
        return ["torchrun", f"--nproc_per_node={n_gpus}", f"--master-port={_free_port()}",
                "-m", "training.trainer", "--config", TRIAL_CONFIG_NAME]
    return [sys.executable, "-m", "training.trainer", "--config", TRIAL_CONFIG_NAME]


def run_training(
    trial_dir: str | Path,
    n_gpus: int,
    log_path: str | Path,
    dry_run: bool = False,
) -> int:
    """Train one trial, reading its config from trial_dir. Returns the exit code."""
    trial_dir = Path(trial_dir)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_command(n_gpus)
    env = os.environ.copy()
    env["MAR_CONFIG_DIR"] = str(trial_dir)

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"cwd={REPO_ROOT}\nMAR_CONFIG_DIR={trial_dir}\n{' '.join(cmd)}\n")
        log.flush()
        if dry_run:
            return 0
        return subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, text=True, cwd=str(REPO_ROOT), env=env
        ).returncode
        
#
