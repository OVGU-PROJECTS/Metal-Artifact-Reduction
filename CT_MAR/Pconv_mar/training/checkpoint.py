"""Checkpoint load/resume + save for training/trainer.py."""
import torch
from training.distributed import unwrap_model

def load_initial_state(model, optimizer, lr_scheduler, ckpt_dir, args, ctx):
    """Restore training state and return the starting "best_val".
    - initial_epoch > 0 resumes last.pt
    - otherwise init_weights loads model weights only
    - empty init_weights starts from scratch
    """
    if args.initial_epoch > 0:
        state = torch.load(ckpt_dir / "last.pt", map_location="cpu")
        unwrap_model(model).load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        if ctx.is_main:
            print(f"[resume] loaded last.pt — resuming from epoch {args.initial_epoch}", flush=True)
        return state["best_val"]

    if args.init_weights:
        state = torch.load(args.init_weights, map_location="cpu")
        unwrap_model(model).load_state_dict(state["model"])
    return float("inf")


def save_checkpoint(ckpt_dir, epoch, model, optimizer, lr_scheduler, weights,
                    best_val, args, world_size, is_best):
    """Save one state dict under up to three names: last, best, periodic."""
    state = {
        "epoch": epoch + 1,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "loss_weights": weights,
        "best_val": best_val,
        "args": vars(args),
        "world_size": world_size,
        "grad_accum_steps": args.grad_accum_steps,
    }
    torch.save(state, ckpt_dir / "last.pt")
    if is_best:
        torch.save(state, ckpt_dir / "best.pt")
    # save_every_epochs=0 disables the periodic dumps
    if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
        torch.save(state, ckpt_dir / f"epoch_{epoch + 1:03d}.pt")
        
