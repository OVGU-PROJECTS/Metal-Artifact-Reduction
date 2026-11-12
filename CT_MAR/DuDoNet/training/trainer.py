"""DuDoNet trainer, single- or multi-GPU (DDP via torchrun).

- each epoch shuffle the cases and shards them
across GPUs without overlap, so no case is repeated in the same epoch.
"""
import math, random, argparse, time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from mar_common.cases import list_case_dirs
from models import DuDoNet2RIL, dudonet_loss_mixed

from .runtime import _load_cfg, ddp_setup, ddp_cleanup
from .artifacts import write_json, csv_init, csv_append, save_ckpt
from .data import load_batch
from .evaluation import evaluate_cases


def train_dudonet(
    *,
    train_root: str,
    val_root: str,
    out_dir: str,
    fov_head_mm: float,
    fov_body_mm: float,
    batch_size: int = 4,
    epochs: int = 380,
    max_epochs_for_smbo: Optional[int] = None,
    trainset: Optional[int] = None,
    valset: Optional[int] = None,
    lr: float = 2e-4,
    seed: int = 42,
    beta1=0.5, beta2=0.999,
    accum_steps: Optional[int] = None,
    lr_step_size: int = 30,
    lr_gamma: float = 0.5,
    save_every: int = 50,
    train_ssim_cases: int = 1000,

):
    """Train DuDoNet end-to-end (SE-Net + RIL + IE-Net) with the L1 paper loss
    - used also for finetuning
    
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # For LR tuning, run a shorter proxy budget instead of the full schedule.
    if max_epochs_for_smbo is not None:
        epochs = min(epochs, int(max_epochs_for_smbo))

    is_ddp, local_rank, rank, world_size = ddp_setup()
    is_main = (rank == 0)

    ckpt_dir = out_dir / "checkpoints"
    metrics_csv = out_dir / "metrics.csv"

    if is_main:
        # Check the metric schema before replacing any existing run metadata.
        csv_init(metrics_csv, header=[
            "epoch","lr","epoch_time_sec",
            "train_L","train_Gs","train_RC","train_Gi",
            "val_L","val_Gs","val_RC","val_Gi",
            "val_ssim_aapm","train_ssim_aapm",
        ])

        # config.json
        write_json(out_dir / "config.json", {
            "train_root": train_root,
            "val_root": val_root,
            "out_dir": str(out_dir),
            "fov_head_mm": fov_head_mm,
            "fov_body_mm": fov_body_mm,
            "epochs": epochs,
            "max_epochs_for_smbo": max_epochs_for_smbo,
            "batch_size": batch_size,
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "lr_step_size": lr_step_size,
            "lr_gamma": lr_gamma,
            "seed": seed,
            "trainset": trainset,
            "valset": valset,
            "accum_steps": accum_steps,
            "save_every": save_every,
            "train_ssim_cases": train_ssim_cases,
            "world_size": world_size,
        })

    # Make randomness different per-rank, but reproducible
    np.random.seed(seed + 10_000 * rank)
    torch.manual_seed(seed + 10_000 * rank)
    torch.cuda.manual_seed_all(seed + 10_000 * rank)

    device = torch.device(f"cuda:{local_rank}") if is_ddp else torch.device("cuda")
    model = DuDoNet2RIL(fov_head_mm, fov_body_mm, img_size=512).to(device)

    if is_ddp:
        # Each process runs on a single GPU: device_ids must be [local_rank]
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step_size, gamma=lr_gamma)

    best_val = float("inf")

    train_cases = list_case_dirs(train_root)
    val_cases   = list_case_dirs(val_root)

    if trainset is None:
        trainset = len(train_cases)
    if valset is None:
        valset = len(val_cases)
    if train_ssim_cases < 1:
        raise ValueError("train_ssim_cases must be at least 1")
    train_ssim_order = list(train_cases)[:trainset]
    random.Random(seed + 50_000).shuffle(train_ssim_order)
    train_ssim_order = train_ssim_order[
        :min(train_ssim_cases, len(train_ssim_order))
    ]

    if accum_steps is None:
        accum_steps = max(1, math.ceil(8 / (batch_size * world_size)))
    else:
        accum_steps = int(accum_steps)
    if accum_steps < 1:
        raise ValueError("accum_steps must be at least 1")

    for epoch in range(epochs):
        t_epoch = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)

        run = {"L": 0.0, "Gs": 0.0, "RC": 0.0, "Gi": 0.0}

        # shuffle all cases identically on every rank
        epoch_order = list(train_cases)
        random.Random(seed + epoch).shuffle(epoch_order)

        # optional subset for SMBO tuning.
        epoch_order = epoch_order[:trainset]

        # DDP: give each rank a disjoint shard so all samples are covered once.
        if is_ddp:
            if len(epoch_order) % world_size != 0:
                raise ValueError(
                    f"trainset={len(epoch_order)} not divisible by world_size={world_size}. "
                    "To use every sample exactly once per epoch in DDP without repetition, "
                    "make trainset divisible by number of GPUs (or run single GPU)."
                )
            epoch_order = epoch_order[rank::world_size]

        # Equal sample count per rank => equal step count => DDP won't hang.
        steps_per_epoch = math.ceil(len(epoch_order) / batch_size)

        for chunk_start in range(0, steps_per_epoch, accum_steps):
            chunk_end = min(chunk_start + accum_steps, steps_per_epoch)
            chunk_steps = chunk_end - chunk_start

            for step in range(chunk_start, chunk_end):
                s = step * batch_size
                case_batch = epoch_order[s : s + batch_size]
                if len(case_batch) == 0:
                    break

                y_li, mt, y_gt, x_gt, fov = load_batch(case_batch, device=device)
                loss, terms = dudonet_loss_mixed(model, y_li, mt, y_gt, x_gt, fov)
                # Divide by the actual chunk size so the final partial chunk is
                (loss / chunk_steps).backward()

                for k in run:
                    run[k] += terms[k]

            opt.step()
            opt.zero_grad(set_to_none=True)

        # Average within this rank (we keep printing/saving only on rank 0).
        run = {k: v / max(steps_per_epoch, 1) for k, v in run.items()}

        eval_model = model.module if is_ddp else model

        # Fixed train/validation SSIM evaluation (rank 0 only).
        if is_main:
            eval_model.eval()
            _, train_ssim = evaluate_cases(
                eval_model, train_ssim_order, batch_size, device,
                calculate_loss=False,
            )
            vrun, val_ssim = evaluate_cases(
                eval_model, list(val_cases)[:valset], batch_size, device,
                calculate_loss=True,
            )

        sched.step()

        if is_ddp:
            t = torch.tensor([run["L"], run["Gs"], run["RC"], run["Gi"]], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= world_size
            run["L"], run["Gs"], run["RC"], run["Gi"] = [float(x) for x in t]

        if is_main:
            epoch_time = time.time() - t_epoch

            # Decide current-best BEFORE saving anything
            is_best = (vrun["L"] < best_val)
            if is_best:
                best_val = float(vrun["L"])

            # log CSV (train is already all-reduced; val is rank0-only)
            csv_append(metrics_csv, [
                epoch + 1,
                float(opt.param_groups[0]["lr"]),
                float(epoch_time),
                float(run["L"]), float(run["Gs"]), float(run["RC"]), float(run["Gi"]),
                float(vrun["L"]), float(vrun["Gs"]), float(vrun["RC"]), float(vrun["Gi"]),
                val_ssim["ssim_aapm"], train_ssim["ssim_aapm"],
            ])

            # save last (always)
            save_ckpt(
                ckpt_dir / "last.pt",
                model, opt, sched,
                epoch=epoch + 1,
                best_val=best_val,
            )

            # save best (only if improved)
            if is_best:
                save_ckpt(
                    ckpt_dir / "best.pt",
                    model, opt, sched,
                    epoch=epoch + 1,
                    best_val=best_val,
                )
            # save periodic snapshot every `save_every` epochs (kept permanently)
            if save_every and (epoch + 1) % save_every == 0:
                save_ckpt(
                    ckpt_dir / f"epoch_{epoch + 1:04d}.pt",
                    model, opt, sched,
                    epoch=epoch + 1,
                    best_val=best_val,
                )


            print(
                f"Epoch {epoch+1:03d}/{epochs} | "
                f"train L={run['L']:.5f} (Gs={run['Gs']:.5f}, RC={run['RC']:.5f}, Gi={run['Gi']:.5f}) | "
                f"val L={vrun['L']:.5f} | "
                f"train AAPM SSIM={train_ssim['ssim_aapm']:.6f} | "
                f"val AAPM SSIM={val_ssim['ssim_aapm']:.6f} | "
                f"lr={opt.param_groups[0]['lr']:.2e} | "
            )


        if is_ddp:
            try:
                dist.barrier()
            except Exception as e:
                if is_main:
                    print("Barrier failed:", str(e))

    ddp_cleanup(is_ddp)
    return best_val if is_main else None  # consumed by Optuna tuning



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dudonet",
                    help="Name of configs/<config>.yaml to load (or $MAR_CONFIG_DIR/<config>.yaml).")
    args = ap.parse_args()
    # All settings come from the YAML; a bad key raises TypeError from train_dudonet.
    train_dudonet(**vars(_load_cfg(args.config)))
