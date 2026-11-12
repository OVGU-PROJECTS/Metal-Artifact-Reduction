#!/usr/bin/env python3

import os
import csv
import time
import json
import copy
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Subset
from torch.optim.lr_scheduler import StepLR

import yaml
from data.patches_generator import PatchesGenerator
from MAR_Inv.vae.network import BigAE
from MAR_Inv.vae.critic import Critic
import MAR_Inv.utils.newtraining_utils as training_utils


# -------------------------
# load the YAML config file (command-line related setup)
# -------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def set_global_seeds(seed: int) -> None:
    """Set RNG seeds for numpy and PyTorch so training is repeatable."""
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_dev_ids(dev_yaml) -> List[int]:
    # parse device ids from YAML/CLI into a simple list of ints
    if isinstance(dev_yaml, int):
        return [int(dev_yaml)]
    if isinstance(dev_yaml, (list, tuple)):
        return [int(x) for x in dev_yaml]
    if isinstance(dev_yaml, str):
        return [int(x.strip()) for x in dev_yaml.split(",") if x.strip() != ""]
    return [0]


def _validate_dev_ids(dev_ids: List[int], device: torch.device) -> List[int]:
    if device.type != "cuda":
        return []
    n_vis = torch.cuda.device_count()
    bad = [d for d in dev_ids if d < 0 or d >= n_vis]
    if bad:
        raise ValueError(
            f"Invalid device ids {bad}. Visible device_count={n_vis} so valid ids are 0..{n_vis-1}. "
            f"Fix YAML devices or CLI --devices, or adjust CUDA_VISIBLE_DEVICES."
        )
    return dev_ids


def subset_dataset(ds, n: Optional[int], seed: int):
    # allow smaller experiments by sampling a random subset of the dataset
    if n is None or int(n) <= 0 or int(n) >= len(ds):
        return ds
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(ds))[: int(n)].tolist()
    return Subset(ds, idx)


def set_dataset_epoch(ds, epoch: int) -> None:
    # some datasets need an epoch number to reseed their sampler each epoch
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(int(epoch))
        return
    base = getattr(ds, "dataset", None)
    if base is not None and hasattr(base, "set_epoch"):
        base.set_epoch(int(epoch))


def json_dump(path: str, obj: Dict[str, Any]) -> None:
    # write JSON atomically so we never leave a partially written file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# -------------------------

def freeze_(module: torch.nn.Module, flag: bool) -> None:
    # Freeze or unfreeze model params during the alternating GAN updates.
    for p in module.parameters():
        p.requires_grad_(not flag)


def gradient_penalty(critic, target, fake, lam=10.0):
    # WGAN-GP penalty: keeps the critic Lipschitz and training stable.
    assert target.size() == fake.size()
    dev = target.device
    a = torch.rand((target.size(0), 1, 1, 1), device=dev)
    interp = (a * target + ((1 - a) * fake)).requires_grad_(True)
    d_interp = critic(interp)
    fake_ = torch.ones_like(d_interp, device=dev, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interp, inputs=interp, grad_outputs=fake_, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * float(lam)
    return gradient_penalty


def _get_conditioning(batch, conditional: bool, device: torch.device):
    # build the extra image conditioning input when needed by the model.
    if not conditional:
        return None
    # conditioning = [pred, mask] => (B,2,H,W)
    if "p" not in batch or "m" not in batch:
        raise KeyError("conditional=True requires batch keys 'p' and 'm'.")
    return torch.cat([batch["p"], batch["m"]], dim=1).to(device).float()


@torch.no_grad()
def eval_epoch(
    *,
    generator,
    critic,
    loader,
    device,
    ae_loss,
    perc_loss,
    alpha: float,
    beta: float,
    kl_weight: float,
    lam: float,
    patchsize: int,
    max_steps: int,
    conditional: bool,
    cond_critic: bool,
    domain: str = "x",
) -> Dict[str, float]:
    # run validation with no weight updates, just gather metrics.
    generator.eval()
    critic.eval()

    acc = {"D": 0.0, "rec": 0.0, "perc": 0.0, "kl": 0.0, "adv": 0.0, "G": 0.0}
    n = 0

    it = iter(loader)
    for _ in range(int(max_steps)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        x = batch[domain].to(device).float()
        im_cond = _get_conditioning(batch, conditional=conditional, device=device)
        kl, recon = generator(x, im_cond=im_cond)

        rec, klv = ae_loss(recon, x, kl, is_kl=True)
        perc = perc_loss(recon, x)

        if cond_critic:
            c_recon = torch.cat([recon, im_cond], dim=1)
            c_x = torch.cat([x, im_cond], dim=1)
        else:
            c_recon = recon
            c_x = x

        adv = -torch.mean(critic(c_recon))
        G = float(alpha) * rec + float(beta) * perc + float(kl_weight) * klv + adv

        # D metric (for logging only; no optimizer step)
        with torch.enable_grad():
            gp = gradient_penalty(critic, c_x.detach(), c_recon.detach(), lam=float(lam))
        critic_loss = torch.mean(critic(c_recon)) - torch.mean(critic(c_x))
        D = critic_loss + gp

        acc["D"] += float(D.detach().cpu())
        acc["rec"] += float(rec.detach().cpu())
        acc["perc"] += float(perc.detach().cpu())
        acc["kl"] += float(klv.detach().cpu())
        acc["adv"] += float(adv.detach().cpu())
        acc["G"] += float(G.detach().cpu())
        n += 1

    for k in acc:
        acc[k] /= max(1, n)
    return acc


def train_epoch(
    *,
    generator,
    critic,
    loader,
    device,
    g_opt,
    d_opt,
    ae_loss,
    perc_loss,
    alpha: float,
    beta: float,
    kl_weight: float,
    lam: float,
    patchsize: int,
    max_steps: int,
    conditional: bool,
    g_sched=None,
    d_sched=None,
    cond_critic: bool,
    domain: str = "x",
) -> Dict[str, float]:
    # run one training epoch with alternating critic/generator updates.
    generator.train()
    critic.train()

    acc = {"D": 0.0, "rec": 0.0, "perc": 0.0, "kl": 0.0, "adv": 0.0, "G": 0.0}
    n = 0

    steps_per_epoch = max(1, len(loader))
    it = iter(loader)

    for step in range(int(max_steps)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        inputs = batch[domain].to(device).float()
        targets = inputs.detach().clone()
        im_cond = _get_conditioning(batch, conditional=conditional, device=device)

        # ---- Critic updates (WGAN-GP) ----
        with torch.no_grad():
            _kl, fakes = generator(inputs, im_cond=im_cond)
        fakes = fakes.detach()
        #  Train Critic
        if cond_critic:
            c_fakes = torch.cat([fakes, im_cond], dim=1)
            c_targets = torch.cat([targets, im_cond], dim=1)
        else:
            c_fakes = fakes
            c_targets = targets

        grad_p = gradient_penalty(critic, c_targets, c_fakes, lam=float(lam))
        critic_loss = torch.mean(critic(c_fakes)) - torch.mean(critic(c_targets))
        loss_D = critic_loss + grad_p

        d_opt.zero_grad(set_to_none=True)
        loss_D.backward()
        d_opt.step()

        # ---- Generator update ----
        freeze_(critic, True)
        g_opt.zero_grad(set_to_none=True)

        kl, fakes = generator(inputs, im_cond=im_cond)
        loss_rec, loss_kl = ae_loss(fakes, targets, kl, is_kl=True)
        loss_perc = perc_loss(fakes, targets)
        
        if cond_critic:
            loss_adv = -torch.mean(critic(torch.cat([fakes, im_cond], dim=1)))
        else:
            loss_adv = -torch.mean(critic(fakes))
            
        loss_G = float(alpha) * loss_rec + float(beta) * loss_perc + float(kl_weight) * loss_kl + loss_adv

        loss_G.backward()
        g_opt.step()
        freeze_(critic, False)

        if (not torch.isfinite(loss_G)) or (not torch.isfinite(loss_D)):
            raise RuntimeError(f"Non-finite loss: G={loss_G.item()} D={loss_D.item()}")

        acc["D"] += float(loss_D.detach().cpu())
        acc["rec"] += float(loss_rec.detach().cpu())
        acc["perc"] += float(loss_perc.detach().cpu())
        acc["kl"] += float(loss_kl.detach().cpu())
        acc["adv"] += float(loss_adv.detach().cpu())
        acc["G"] += float(loss_G.detach().cpu())
        n += 1

        # StepLR is epoch-based → step at epoch boundary
        if ((step + 1) % steps_per_epoch) == 0:
            if g_sched is not None:
                g_sched.step()
            if d_sched is not None:
                d_sched.step()

    for k in acc:
        acc[k] /= max(1, n)
    return acc


# -------------------------
# Trainer
# -------------------------
class Trainer:
    def __init__(self, cfg: Dict[str, Any], args: argparse.Namespace, device: torch.device, dev_ids: List[int], out_dir: str):
        self.cfg = cfg
        self.args = args
        self.device = device
        self.dev_ids = dev_ids

        self.out_dir = out_dir
        self.ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # data paths and general config options
        train_root = cfg.get("train_datafolder")
        val_root = cfg.get("val_datafolder")
        if not train_root or not val_root:
            raise KeyError("Set train_datafolder/val_datafolder (or datafolder) in YAML.")
        stats_path = cfg["stats"]

        fn_x = cfg.get("fn_x", "input_img.mha")
        fn_y = cfg.get("fn_y", "output_img.mha")
        self.conditional = bool(cfg.get("conditional", True))

        self.patchsize =  int(cfg["patchsize"])
        train_n = int(cfg.get("train_n", 0))
        val_n = int(cfg.get("val_n", 0))
        seeds = int(cfg["seed"])
        
        # fixed scalars (same as eval)
        self.alpha = float(cfg["alpha"])
        self.beta = float(cfg["beta"])
        self.kl_weight = float(cfg["kl_weight"])
        self.lam = float(cfg["lam"])

        # epoch plan
        self.epochs = int(cfg["epochs"])
        self.train_steps = int(cfg.get("train_steps", 0))
        self.val_steps = int(cfg.get("val_steps", 0))

        self.best_val_rec = float("inf")
        self.best_epoch = -1

        base_args_train = argparse.Namespace(**cfg)
        base_args_train.datafolder = train_root

        base_args_val = copy.deepcopy(base_args_train)
        base_args_val.datafolder = val_root

        # build the train/val datasets used by the project pipeline
        train_ds_full = PatchesGenerator(
            base_args_train,
            fn_x=fn_x,
            fn_y=fn_y,
            patches_per_image=int(cfg.get("patches_per_image")),
            strong_ratio=float(cfg.get("strong_ratio")),
            topk=int(cfg.get("topk")),
            use_strength=bool(cfg.get("use_strength")),
            return_meta=False,
            stats_path=stats_path,
            conditional=self.conditional,
            require_cond_files=self.conditional,
        )

        val_ds_full = PatchesGenerator(
            base_args_val,
            fn_x=fn_x,
            fn_y=fn_y,
            patches_per_image=int(cfg["patches_per_image"]) if "patches_per_image" in cfg else 1,
            strong_ratio=float(cfg["strong_ratio"]) if "strong_ratio" in cfg else 0.7,
            topk=int(cfg["topk"]) if "topk" in cfg else 5000,
            use_strength=bool(cfg["use_strength"]) if "use_strength" in cfg else True,
            return_meta=False,
            stats_path=stats_path,
            conditional=self.conditional,
            require_cond_files=self.conditional,
        )

        # optionally subsample the datasets for debugging or smaller experiments
        self.train_ds = subset_dataset(train_ds_full, train_n, seed=seeds) if train_n > 0 else train_ds_full
        self.val_ds   = subset_dataset(val_ds_full, val_n, seed=seeds) if val_n > 0 else val_ds_full

        loaders = training_utils.setup_dataloader(args, {"train": self.train_ds, "val": self.val_ds})
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]

        if len(self.train_loader) == 0:
            raise RuntimeError("train_loader is empty (len==0). Check drop_last/mbs/train_n/dataset.")
        if self.train_steps <= 0:
            self.train_steps = len(self.train_loader)
        if self.val_steps <= 0:
            self.val_steps = len(self.val_loader)

        # build the generator and critic used in this project setup
        self.cond_ch = 2 if self.conditional else 0
        self.cond_critic = bool(cfg.get("cond_critic", False))
        if self.cond_critic and not self.conditional:
            raise ValueError("cond_critic=True requires conditional=True.")
        self.critic_in_ch = (1 + self.cond_ch) if self.cond_critic else 1
        
        self.generator = BigAE(args, in_ch=1+ self.cond_ch, out_ch=1, cond_ch=self.cond_ch).to(device)
        self.critic = Critic(args, in_ch=self.critic_in_ch, input_ft=64, depth=5, max_ft=512, norm="in").to(device)

        if device.type == "cuda" and len(dev_ids) > 1:
            self.generator = torch.nn.DataParallel(self.generator, device_ids=dev_ids)
            self.critic = torch.nn.DataParallel(self.critic, device_ids=dev_ids)

        # optimizers
        self.g_opt = training_utils.setup_optimizer(args, self.generator.parameters())

        self.d_opt = optim.Adam(
            self.critic.parameters(),
            lr=float(cfg["lr"]) * float(cfg.get("d_lr_mult")),
            betas=(float(cfg["adam_b1"]), float(cfg["adam_b2"])),
        )

        # schedulers (ONLY if lr_decay true; keys must exist like eval code expectation)
        self.g_sched = None
        self.d_sched = None
        if bool(cfg["lr_decay"]):
            if "lr_decay_gamma" not in cfg or "lr_decay_step_epochs" not in cfg:
                raise KeyError("lr_decay is true, but lr_decay_gamma and/or lr_decay_step_epochs missing in YAML.")
            self.g_sched = StepLR(self.g_opt, step_size=int(cfg["lr_decay_step_epochs"]), gamma=float(cfg["lr_decay_gamma"]))
            self.d_sched = StepLR(self.d_opt, step_size=int(cfg["lr_decay_step_epochs"]), gamma=float(cfg["lr_decay_gamma"]))

        # losses
        self.ae_loss = training_utils.AELoss(args)
        self.perc_loss = training_utils.PerceptualLoss(args).to(device).eval()

    
        # CSV logger for tracking train/val metrics per epoch
        self.csv_path = os.path.join(out_dir, "losses.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch",
                    "train_D", "train_rec", "train_perc", "train_kl", "train_adv", "train_G",
                    "val_D", "val_rec", "val_perc", "val_kl", "val_adv", "val_G",
                    "g_lr_end", "d_lr_end",
                ])

    def _ckpt_state(self, epoch: int) -> Dict[str, Any]:
        gen_state = self.generator.module.state_dict() if hasattr(self.generator, "module") else self.generator.state_dict()
        crit_state = self.critic.module.state_dict() if hasattr(self.critic, "module") else self.critic.state_dict()

        state = {
            "epoch": int(epoch),
            "generator_state_dict": gen_state,
            "critic_state_dict": crit_state,
            "g_optimizer_state_dict": self.g_opt.state_dict(),
            "d_optimizer_state_dict": self.d_opt.state_dict(),
            "config": dict(self.cfg),
        }

        if self.g_sched is not None:
            state["g_scheduler_state_dict"] = self.g_sched.state_dict()
        if self.d_sched is not None:
            state["d_scheduler_state_dict"] = self.d_sched.state_dict()

        return state

    def save_checkpoint(self, epoch: int, val_rec: float) -> str:
        state = self._ckpt_state(epoch)
        saved = []

        # 1) always save last
        last_path = os.path.join(self.ckpt_dir, "last.pt")
        self._atomic_torch_save(state, last_path)
        saved.append("last.pt")

        # 2) save best if val_rec improved
        if float(val_rec) < self.best_val_rec:
            self.best_val_rec = float(val_rec)
            self.best_epoch = int(epoch)

            best_state = dict(state)
            best_state["best_val_rec"] = self.best_val_rec
            best_state["best_epoch"] = self.best_epoch

            best_path = os.path.join(self.ckpt_dir, "best.pt")
            self._atomic_torch_save(best_state, best_path)
            saved.append("best.pt")

        # 3) save numbered checkpoint every 50 epochs and at final epoch
        if ((epoch + 1) % 50 == 0) or (epoch == int(self.epochs) - 1):
            epoch_path = os.path.join(self.ckpt_dir, f"epoch_{epoch+1:04d}.pt")
            self._atomic_torch_save(state, epoch_path)
            saved.append(os.path.basename(epoch_path))

        return ", ".join(saved)

    def fit(self):
        # save run metadata once so we can reproduce the experiment later
        json_dump(os.path.join(self.out_dir, "run_meta.json"), {
            "created": datetime.now().isoformat(),
            "device": str(self.device),
            "devices": list(self.dev_ids),
            "epochs": int(self.epochs),
            "train_steps_per_epoch": int(self.train_steps),
            "val_steps_per_epoch": int(self.val_steps),
            "len_train_loader": int(len(self.train_loader)),
            "len_val_loader": int(len(self.val_loader)),
            "objective": "val_rec",
        })

        # main epoch loop: train, evaluate, log, save checkpoints
        for epoch in range(int(self.epochs)):
            set_global_seeds(int(self.cfg["seed"]) + int(epoch))
            set_dataset_epoch(self.train_ds, epoch)
            set_dataset_epoch(self.val_ds, epoch)

            t0 = time.perf_counter()

            tr = train_epoch(
                generator=self.generator,
                critic=self.critic,
                loader=self.train_loader,
                device=self.device,
                g_opt=self.g_opt,
                d_opt=self.d_opt,
                ae_loss=self.ae_loss,
                perc_loss=self.perc_loss,
                alpha=self.alpha,
                beta=self.beta,
                kl_weight=self.kl_weight,
                lam=self.lam,
                patchsize=self.patchsize,
                max_steps=self.train_steps,
                conditional=self.conditional,
                g_sched=self.g_sched,
                d_sched=self.d_sched,
                cond_critic=self.cond_critic,
                domain=getattr(self.args, "domain", "x"),
            )

            va = eval_epoch(
                generator=self.generator,
                critic=self.critic,
                loader=self.val_loader,
                device=self.device,
                ae_loss=self.ae_loss,
                perc_loss=self.perc_loss,
                alpha=self.alpha,
                beta=self.beta,
                kl_weight=self.kl_weight,
                lam=self.lam,
                patchsize=self.patchsize,
                max_steps=self.val_steps,
                conditional=self.conditional,
                cond_critic=self.cond_critic,
                domain=getattr(self.args, "domain", "x"),
            )

            g_lr = float(self.g_opt.param_groups[0]["lr"])
            d_lr = float(self.d_opt.param_groups[0]["lr"])

            # CSV append
            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    int(epoch),
                    tr["D"], tr["rec"], tr["perc"], tr["kl"], tr["adv"], tr["G"],
                    va["D"], va["rec"], va["perc"], va["kl"], va["adv"], va["G"],
                    g_lr, d_lr
                ])

            ckpt_path = self.save_checkpoint(epoch, va["rec"])

            dt = time.perf_counter() - t0
            print(
                
                f"[epoch {epoch:03d}/{int(self.epochs)-1:03d}] "
                f"train: rec={tr['rec']:.4f} perc={tr['perc']:.4f} kl={tr['kl']:.4f} D={tr['D']:.4f} G={tr['G']:.4f} | "
                f"val: rec={va['rec']:.4f} perc={va['perc']:.4f} kl={va['kl']:.4f} D={va['D']:.4f} G={va['G']:.4f} | "
                f"g_lr={g_lr:.3e} d_lr={d_lr:.3e} | "
                f"time={dt:.1f}s | ckpt={os.path.basename(ckpt_path)}"
            )
    def _atomic_torch_save(self, obj: Dict[str, Any], path: str) -> None:
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)


# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    ap.add_argument("--devices", type=str, default="", help="Override devices, e.g. 0,1,2 (else use YAML).")
    ap.add_argument("--out_dir", type=str, default="", help="Output folder (default: ./runs/<timestamp>)")
    args_cli = ap.parse_args()

    cfg = load_yaml(args_cli.config)

    # STRICT required keys (no hidden defaults)
    required = [
        # data
        "stats",
        # loop control
        "epochs", "seed",
        # dataloader
        "mbs", "num_workers",
        # model/loss/gan
        "patchsize",
        "alpha", "beta", "kl_weight",
        "lam",
        # optimizer
        "optimizer", "lr", "adam_b1", "adam_b2",
        # devices + lr_decay switch
        "devices", "lr_decay",
        # criterion expected by AELoss
        "criterion",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required YAML keys: {missing}")

    # CLI override of devices (same behavior as eval code)
    if args_cli.devices.strip():
        cfg["devices"] = [int(x.strip()) for x in args_cli.devices.split(",") if x.strip() != ""]

    # device + ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_ids = _normalize_dev_ids(cfg["devices"])
    dev_ids = _validate_dev_ids(dev_ids, device)
    cfg["devices"] = dev_ids

    # build args exactly like eval code
    args = argparse.Namespace(**cfg)
    args.cuda = (device.type == "cuda")

    # choose output folder for all logs/checkpoints
    if args_cli.out_dir.strip():
        out_dir = args_cli.out_dir
    else:
        out_dir = os.path.join("./runs", f"vae_train_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    # global seed once (epoch loop will add +epoch)
    set_global_seeds(int(cfg["seed"]))

    print("CUDA available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
    print(f"[info] device = {device}")
    print(f"[info] devices = {dev_ids}")
    print(f"[info] out_dir = {out_dir}")

    trainer = Trainer(cfg=cfg, args=args, device=device, dev_ids=dev_ids, out_dir=out_dir)
    print(f"[info] train_steps={trainer.train_steps} val_steps={trainer.val_steps} epochs={int(cfg['epochs'])}")
    trainer.fit()

    print("\n✅ training done")


if __name__ == "__main__":
    main()
