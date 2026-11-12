#!/usr/bin/env python3
"""Train a DML embedding for MAR invariance evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from MAR_Inv.dml.dataset import DMLSampleBankDataset
from MAR_Inv.dml.losses import semihard_triplet_loss
from MAR_Inv.dml.network import MetricResNet


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def run_epoch(model, loader, optimizer, device, margin: float, train: bool) -> Dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "pos_dist": 0.0, "neg_dist": 0.0, "active_frac": 0.0, "semihard_frac": 0.0}
    n = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if train else "val"):
            anchor = batch["anchor"].to(device).float()
            positive = batch["positive"].to(device).float()
            labels = batch["label"].to(device).long()
            if int(anchor.shape[0]) < 2:
                continue
            if int(torch.unique(labels).numel()) < 2:
                continue

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            loss, stats = semihard_triplet_loss(anchor_emb, positive_emb, margin=margin, labels=labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            for key in totals:
                if key != "loss":
                    totals[key] += float(stats[key].detach().cpu())
            n += 1

    for key in totals:
        totals[key] /= max(1, n)
    return totals


def save_checkpoint(path: str, model, optimizer, cfg: Dict[str, Any], epoch: int, val_loss: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": int(epoch),
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": dict(cfg),
        "val_loss": float(val_loss),
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def build_dml_dataset(
    sample_bank,
    *,
    max_items: int = 0,
    positive_index=None,
    samples_per_case: int = 0,
    randomize_samples: bool = False,
):
    if isinstance(sample_bank, (list, tuple)):
        return ConcatDataset(
            [
                DMLSampleBankDataset(
                    str(path),
                    max_items=max_items,
                    positive_index=positive_index,
                    samples_per_case=samples_per_case,
                    randomize_samples=randomize_samples,
                )
                for path in sample_bank
            ]
        )
    return DMLSampleBankDataset(
        str(sample_bank),
        max_items=max_items,
        positive_index=positive_index,
        samples_per_case=samples_per_case,
        randomize_samples=randomize_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--devices", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    cli = parser.parse_args()

    cfg = load_yaml(cli.config)
    required = ["sample_bank", "epochs", "mbs", "lr", "adam_b1", "adam_b2", "margin", "seed"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError(f"Missing DML config keys: {missing}")

    if cli.devices.strip():
        cfg["devices"] = [int(x.strip()) for x in cli.devices.split(",") if x.strip()]
    cfg.setdefault("devices", [0])
    cfg.setdefault("num_workers", 0)
    cfg.setdefault("embedding_dim", 128)
    cfg.setdefault("backbone", "resnet50")
    cfg.setdefault("pretrained", True)
    cfg.setdefault("dropout", 0.0)
    cfg.setdefault("weight_decay", 0.0)
    cfg.setdefault("max_train_items", 0)
    cfg.setdefault("max_val_items", 0)
    cfg.setdefault("train_samples_per_case", 2)
    cfg.setdefault("val_samples_per_case", 2)
    cfg.setdefault("train_randomize_samples", False)
    cfg.setdefault("lr_step_size", 10)
    cfg.setdefault("lr_gamma", 0.7)

    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cli.out_dir.strip() or os.path.join("./runs", f"dml_train_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    train_ds = build_dml_dataset(
        cfg["sample_bank"],
        max_items=int(cfg["max_train_items"]),
        samples_per_case=int(cfg["train_samples_per_case"]),
        randomize_samples=bool(cfg["train_randomize_samples"]),
    )
    val_bank = cfg.get("val_sample_bank", cfg["sample_bank"])
    val_ds = build_dml_dataset(
        val_bank,
        max_items=int(cfg["max_val_items"]),
        samples_per_case=int(cfg["val_samples_per_case"]),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["mbs"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["mbs"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty. Lower mbs or increase sample bank size.")

    args = argparse.Namespace(**cfg)
    model = MetricResNet.from_args(args).to(device)
    dev_ids = [int(x) for x in cfg["devices"]]
    if device.type == "cuda" and len(dev_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=dev_ids)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        betas=(float(cfg["adam_b1"]), float(cfg["adam_b2"])),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg["lr_step_size"]),
        gamma=float(cfg["lr_gamma"]),
    )

    csv_path = os.path.join(out_dir, "losses.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_pos_dist",
            "train_neg_dist",
            "train_active_frac",
            "train_semihard_frac",
            "val_loss",
            "val_pos_dist",
            "val_neg_dist",
            "val_active_frac",
            "val_semihard_frac",
        ])

    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump({"config": cfg, "device": str(device)}, f, indent=2)

    best_val = float("inf")
    for epoch in range(int(cfg["epochs"])):
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(model, train_loader, optimizer, device, float(cfg["margin"]), train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, float(cfg["margin"]), train=False)
        scheduler.step()

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"],
                train_metrics["pos_dist"],
                train_metrics["neg_dist"],
                train_metrics["active_frac"],
                train_metrics["semihard_frac"],
                val_metrics["loss"],
                val_metrics["pos_dist"],
                val_metrics["neg_dist"],
                val_metrics["active_frac"],
                val_metrics["semihard_frac"],
            ])

        save_checkpoint(os.path.join(out_dir, "checkpoints", "last.pt"), model, optimizer, cfg, epoch, val_metrics["loss"])
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(os.path.join(out_dir, "checkpoints", "best.pt"), model, optimizer, cfg, epoch, best_val)

        print(
            f"[epoch {epoch:03d}] "
            f"lr={epoch_lr:.2e} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"pos={val_metrics['pos_dist']:.4f} neg={val_metrics['neg_dist']:.4f}"
        )


if __name__ == "__main__":
    main()
'''
GPU1=$(gpurequest)

CUDA_VISIBLE_DEVICES="$GPU1" python -u -m MAR_Inv.dml.training \
  --config configs/dml.yaml \
  --out_dir "results/runs/evaluation/dml_new"

'''
