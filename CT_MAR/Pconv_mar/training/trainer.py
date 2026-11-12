"""Training entry point.
- Run via train.sh with torchrun for multi-GPU DDP.
- Wires together:
data loaders, PConvUNet, the calibrated+scheduled CombinedLoss,
Adam with an exponential LR decay, gradient accumulation,
checkpointing, and CSV/TensorBoard logging.
All hyperparameters come from configs/pconv.yaml.
"""
import argparse, json, math, random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from model.loss import CombinedLoss, LossWeightScheduler
from model.pconv_model import PConvUNet
from data.dataset import build_loader
from training.config import load_config
from training.distributed import (setup_distributed, is_distributed,
                                  reduce_mean, unwrap_model, to_device)
from training.evaluation import evaluate, evaluate_breakdown
from training.checkpoint import load_initial_state, save_checkpoint
from training.logging_utils import init_logs, write_epoch_row, write_breakdown, write_tb_epoch

def seed_all(args, ctx):
    """every rank gets its own seed, so the ranks don't draw the same samples"""
    seed = int(args.seed) + ctx.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_stats(args):
    """the head/body mean and std the sinograms were normalized with"""
    with open(args.stats_path, "r") as f:
        stats = json.load(f)
    return ((stats["head"]["mean"], stats["body"]["mean"]),
            (stats["head"]["std"], stats["body"]["std"]))


def wire_data(args, gm, gs, ctx):
    """Build the train, val"""
    common = dict(batch_size=args.batch_size, global_mean=gm, global_std=gs,seed=args.seed)
    
    train_loader, train_sampler = build_loader(
        base_dir=args.train_root, shuffle=True,
        num_workers=args.num_workers, distributed=ctx.distributed,
        rank=ctx.rank, world_size=ctx.world_size, **common,
    )
    val_loader, _ = build_loader(
        base_dir=args.val_root, shuffle=False,
        num_workers=args.num_workers, distributed=ctx.distributed,
        rank=ctx.rank, world_size=ctx.world_size, **common,
    )
    # the breakdown loaders use a deterministic shuffled case order, so the same
    # random subset is scored every epoch
    val_bd_loader, _ = build_loader(
        base_dir=args.val_root, shuffle=False,
        num_workers=0, fixed_case_shuffle=True,
        fixed_case_seed=args.seed + 2000, distributed=False,
        rank=0, world_size=1, **common,
    )
    train_bd_loader, _ = build_loader(
        base_dir=args.train_root,shuffle=False,
        num_workers=0, fixed_case_shuffle=True,
        fixed_case_seed=args.seed + 1000, distributed=False,
        rank=0, world_size=1, **common,
    )
    return train_loader, train_sampler, val_loader, val_bd_loader, train_bd_loader


def wire_model(args, gm, gs, ctx):
    """PConvUNet with its calibration scales, then the loss, optimizer and LR schedule."""
    model = PConvUNet(global_mean=gm, global_std=gs).to(ctx.device)
    model.set_calibration_scales(args.rmse_scale, args.hole_scale, args.dc_scale,
                                 args.img_scale, args.edge_scale, args.perc_scale)
    if ctx.distributed:
        model = DDP(model, device_ids=[ctx.local_rank], output_device=ctx.local_rank)

    scheduler_weights = LossWeightScheduler(
        start_ep_image=args.start_ep_image, start_ep_edge=args.start_ep_edge,
        start_ep_hole=args.start_ep_hole, start_ep_dc=args.start_ep_dc,
        start_ep_perc=args.start_ep_perc,
        ramp_duration=args.ramp_duration, ramp_duration_dc=args.ramp_duration_dc,
        weight_rmse=args.w_rmse, weight_image=args.w_image, weight_edge=args.w_edge,
        weight_hole=args.w_hole, weight_dc=args.w_dc, weight_perc=args.w_perc,
    )
    loss_fn = CombinedLoss(scheduler_weights, unwrap_model(model), dc_mode=args.dc_mode)
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_scheduler = LambdaLR(optimizer, lambda step: 0.9 ** (step / max(1, args.decay_steps)))
    return model, scheduler_weights, loss_fn, optimizer, lr_scheduler

# logs: loss terms breakdowns
def log_breakdowns(logs, model, loss_fn, val_bd_loader, train_bd_loader, ctx, args, epoch):
    """Per-term val+train breakdown (CSV + TensorBoard), every bd_every_epochs only."""
    if (epoch + 1) % args.bd_every_epochs != 0:
        return
    eval_model = unwrap_model(model)
    bd_val = evaluate_breakdown(eval_model, loss_fn, val_bd_loader, ctx.device, max_steps=args.bd_val_steps)
    write_breakdown(bd_val, logs.bd_val_csv, logs.tb_writer, "val", epoch)
    bd_train = evaluate_breakdown(eval_model, loss_fn, train_bd_loader, ctx.device, max_steps=args.bd_train_steps)
    write_breakdown(bd_train, logs.bd_train_csv, logs.tb_writer, "train", epoch)


# training loop: single epoch
def train_one_epoch(model, loss_fn, optimizer, lr_scheduler, train_iter, train_loader,
                    steps_per_epoch, epoch, ctx, args):
    """ run one epoch over a fixed "steps_per_epoch";returns (train_iter, loss_sum)."""
    train_loss_sum = 0.0
    accum_micro_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for step_idx in range(steps_per_epoch):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        (proj, mask), y_true = to_device(batch, ctx.device)
        y_pred = model(proj, mask)
        loss = loss_fn(y_true, y_pred).mean()

        step_loss = loss.item()
        train_loss_sum += step_loss
        accum_micro_batches += 1

        # Gradient accumulation: only step the optimizer every grad_accum_steps
        # micro-batches. Between steps, skip DDP's gradient all-reduce (no_sync) to
        # avoid syncing partial gradients across GPUs.
        should_step = ((step_idx + 1) % args.grad_accum_steps == 0) or ((step_idx + 1) == steps_per_epoch)
        if isinstance(model, DDP) and not should_step:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

        if should_step:
            scale = 1.0 / float(accum_micro_batches)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum_micro_batches = 0

    return train_iter, train_loss_sum

# config, distributed setup for multi GPUs, and the epoch loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pconv",
                    help="Name of configs/<config>.yaml to load (or $MAR_CONFIG_DIR/<config>.yaml).")
    args = load_config(ap.parse_args().config)

    ctx = setup_distributed(args.n_gpus)

    seed_all(args, ctx)

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    csv_dir = run_dir / "csv"
    logs_dir = run_dir / "logs"
    tb_dir = run_dir / "tb"
    if ctx.is_main:
        for d in [run_dir, ckpt_dir, csv_dir, logs_dir, tb_dir]:
            d.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    if is_distributed():
        dist.barrier()

    gm, gs = read_stats(args)

    (train_loader, train_sampler, val_loader,
     val_bd_loader, train_bd_loader) = wire_data(args, gm, gs, ctx)

    global_batch_size = args.batch_size * ctx.world_size
    steps_per_epoch = math.ceil(args.n_train / global_batch_size)
    val_steps = math.ceil(args.n_val / global_batch_size)

    (model, scheduler_weights, loss_fn,
     optimizer, lr_scheduler) = wire_model(args, gm, gs, ctx)

    best_val = load_initial_state(model, optimizer, lr_scheduler, ckpt_dir, args, ctx)

    logs = init_logs(csv_dir, logs_dir, tb_dir, args.initial_epoch) if ctx.is_main else None

    # epoch-0 baseline row, before any weight update
    if args.initial_epoch == 0 and ctx.is_main:
        log_breakdowns(logs, model, loss_fn, val_bd_loader, train_bd_loader, ctx, args, epoch=-1)
    if is_distributed():
        dist.barrier()

    train_iter = iter(train_loader)

    try:
        for epoch in range(args.initial_epoch, args.epochs):
            scheduler_weights.step_epoch(epoch)
            model.train()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_iter, train_loss_sum = train_one_epoch(
                model, loss_fn, optimizer, lr_scheduler, train_iter, train_loader,
                steps_per_epoch, epoch, ctx, args,
            )

            train_loss = reduce_mean(train_loss_sum, float(steps_per_epoch), ctx.device)
            val_loss = evaluate(model, loss_fn, val_loader, ctx.device, max_steps=val_steps)
            lr = float(optimizer.param_groups[0]["lr"])
            weights = scheduler_weights.loss_weights

            if ctx.is_main:
                write_epoch_row(logs.csv_path, epoch, train_loss, val_loss, lr, weights)
                log_breakdowns(logs, model, loss_fn, val_bd_loader, train_bd_loader, ctx, args, epoch)
                write_tb_epoch(logs.tb_writer, epoch, train_loss, val_loss, lr, weights)

                is_best = val_loss < best_val
                if is_best:
                    best_val = val_loss

                save_checkpoint(ckpt_dir, epoch=epoch, model=model, optimizer=optimizer,
                                lr_scheduler=lr_scheduler, weights=weights, best_val=best_val,
                                args=args, world_size=ctx.world_size, is_best=is_best)
                print(f"Epoch {epoch + 1:03d} | train_loss={train_loss:.6f} | "
                      f"val_loss={val_loss:.6f} | lr={lr:.6e} | weights={weights} | "
                      f"world_size={ctx.world_size} | accum={args.grad_accum_steps}", flush=True)
            if is_distributed():
                dist.barrier()
    finally:
        if logs is not None:
            logs.tb_writer.close()
        if is_distributed():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
