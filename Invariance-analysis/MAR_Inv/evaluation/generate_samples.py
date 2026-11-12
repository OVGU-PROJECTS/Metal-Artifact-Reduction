#!/usr/bin/env python3
"""Generate precomputed invariance sample banks for VAE/DML evaluation.

The output file stores:
  x:       [N,1,H,W] original corrupted patches
  samples: [N,K,1,H,W] invariance reconstructions for each patch
  p, m:    optional conditioning patches for inspection
"""

from __future__ import annotations

import argparse
import os
import random
from argparse import Namespace
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from MAR_Inv.cinn.network import ConditionalTransformer
from data.patches_generator import PatchesGenerator
from MAR_Inv.vae.network import BigAE


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def move_state_to_model(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Load checkpoints across DataParallel and SpectralNorm wrapper variants.

    Older checkpoints in this project store wrapped decoder parameters as, for
    example, ``conv0.weight_bar``.  The current network implementation may
    expose the same parameter as ``conv0.module.weight_bar``.  This function
    resolves that naming-only difference while still requiring every model
    tensor to be loaded with a matching shape.
    """
    target_state = model.state_dict()
    adapted_state: Dict[str, torch.Tensor] = {}
    unresolved = []

    for raw_key, value in state.items():
        # Remove only a leading DataParallel prefix.  Do not remove a nested
        # ``.module.`` that belongs to a SpectralNorm wrapper.
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key

        candidates = [key]

        # Old wrapper layout:  layer.weight_bar
        # New wrapper layout:  layer.module.weight_bar
        if "." in key:
            parent, leaf = key.rsplit(".", 1)
            candidates.append(f"{parent}.module.{leaf}")

        # Also support loading a newer checkpoint into an older wrapper layout.
        if ".module." in key:
            candidates.append(key.replace(".module.", ".", 1))

        matched_key = None
        for candidate in candidates:
            if candidate not in target_state:
                continue
            if tuple(target_state[candidate].shape) != tuple(value.shape):
                continue
            matched_key = candidate
            break

        if matched_key is None:
            unresolved.append((raw_key, tuple(value.shape), candidates))
            continue

        if matched_key in adapted_state:
            raise RuntimeError(
                f"Checkpoint keys map more than once to model key {matched_key!r}."
            )
        adapted_state[matched_key] = value

    missing = sorted(set(target_state) - set(adapted_state))
    if unresolved or missing:
        details = []
        if unresolved:
            preview = "; ".join(
                f"{key} {shape}" for key, shape, _ in unresolved[:10]
            )
            details.append(
                f"unresolved checkpoint keys ({len(unresolved)}): {preview}"
            )
        if missing:
            details.append(
                f"missing model keys ({len(missing)}): {', '.join(missing[:10])}"
            )
        raise RuntimeError(
            "Checkpoint is not compatible with the instantiated model after "
            "known key-name conversions. " + " | ".join(details)
        )

    model.load_state_dict(adapted_state, strict=True)


def build_models(args: argparse.Namespace, device: torch.device):
    # Load checkpoints on CPU first. The training checkpoints also contain
    # optimizer states, which needlessly consume GPU memory during inference.
    vae_ckpt = torch.load(args.vae_checkpoint, map_location="cpu", weights_only=False)
    vae_args = Namespace(**vae_ckpt["config"])
    ae = BigAE(vae_args, in_ch=3, out_ch=1, cond_ch=2)
    move_state_to_model(ae, vae_ckpt["generator_state_dict"])
    del vae_ckpt
    ae.to(device)
    ae.eval()
    for param in ae.parameters():
        param.requires_grad_(False)

    cinn_ckpt = torch.load(args.cinn_checkpoint, map_location="cpu", weights_only=False)
    cinn_args = Namespace(**cinn_ckpt.get("args", {}))
    model = ConditionalTransformer(cinn_args)
    move_state_to_model(model, cinn_ckpt["model_state_dict"])
    del cinn_ckpt
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return ae, model


def build_dataset(args: argparse.Namespace) -> PatchesGenerator:
    data_args = Namespace(**vars(args))
    data_args.datafolder = args.datafolder
    return PatchesGenerator(
        data_args,
        fn_x=getattr(args, "fn_x", "input_img.mha"),
        fn_y=getattr(args, "fn_y", "output_img.mha"),
        fn_mask=getattr(args, "fn_mask", "input_mask.mha"),
        fn_pred=getattr(args, "fn_pred", None),
        patches_per_image=int(getattr(args, "patches_per_image", 1)),
        strong_ratio=float(getattr(args, "strong_ratio", 0.7)),
        topk=int(getattr(args, "topk", 5000)),
        use_strength=bool(getattr(args, "use_strength", True)),
        return_meta=True,
        stats_path=args.stats,
        conditional=True,
        require_cond_files=True,
    )


def find_case_dirs_recursive(root: str, required_files: List[str]) -> List[str]:
    """Find case directories below root that contain all required files."""
    case_dirs: List[str] = []
    for current_root, _, files in os.walk(root):
        names = set(files)
        if all(name in names for name in required_files):
            case_dirs.append(os.path.abspath(current_root))
    return sorted(case_dirs)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Optional YAML config.")
    parser.add_argument("--vae_checkpoint", type=str, default=None)
    parser.add_argument("--cinn_checkpoint", type=str, default=None)
    parser.add_argument("--datafolder", type=str, default=None)
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--pred_name", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--mbs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--patchsize", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=None,
        help="Seed used to select cases when --max_items is smaller than the dataset.",
    )
    parser.add_argument("--patches_per_image", type=int, default=None)
    parser.add_argument("--strong_ratio", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    strength_group = parser.add_mutually_exclusive_group()
    strength_group.add_argument("--use_strength", dest="use_strength", action="store_true")
    strength_group.add_argument("--no_use_strength", dest="use_strength", action="store_false")
    parser.set_defaults(use_strength=None)
    parser.add_argument("--fn_x", type=str, default=None)
    parser.add_argument("--fn_y", type=str, default=None)
    parser.add_argument("--fn_mask", type=str, default=None)
    parser.add_argument("--fn_pred", type=str, default=None)
    parser.add_argument(
        "--recursive_cases",
        action="store_true",
        help=(
            "Search below --datafolder for case directories. Use this for "
            "layouts such as test_images_1000/head/... and body/..."
        ),
    )
    cli = parser.parse_args()

    cfg = load_yaml(cli.config) if cli.config else {}
    merged = dict(cfg)
    for key, value in vars(cli).items():
        if value is not None:
            merged[key] = value
    defaults = {
        "pred_name": "cnnmar",
        "devices": "0",
        "num_samples": 1000,
        "max_items": 0,
        "mbs": 16,
        "num_workers": 0,
        "patchsize": 128,
        "seed": 1332,
        "patches_per_image": 1,
        "strong_ratio": 0.7,
        "topk": 5000,
        "use_strength": True,
        "fn_x": "input_img.mha",
        "fn_y": "output_img.mha",
        "fn_mask": "input_mask.mha",
    }
    for key, value in defaults.items():
        merged.setdefault(key, value)
    if merged.get("selection_seed") is None:
        merged["selection_seed"] = int(merged["seed"])
    args = Namespace(**merged)

    required = ["vae_checkpoint", "cinn_checkpoint", "datafolder", "stats"]
    missing = [key for key in required if not getattr(args, key, "")]
    if missing:
        raise KeyError(f"Missing required sample generation settings: {missing}")

    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae, cinn = build_models(args, device)
    dataset = build_dataset(args)
    if bool(getattr(args, "recursive_cases", False)):
        required_files = [
            str(args.fn_x),
            str(args.fn_mask),
            str(args.fn_pred) if getattr(args, "fn_pred", None) else f"pred_{args.pred_name}_img.mha",
        ]
        dataset.case_dirs = find_case_dirs_recursive(args.datafolder, required_files)
        if not dataset.case_dirs:
            raise ValueError(
                "No recursive case directories found containing: "
                + ", ".join(required_files)
            )
        print(
            f"recursive case discovery found {len(dataset.case_dirs)} "
            f"case directories below {args.datafolder}"
        )
    total_cases = len(dataset.case_dirs)
    if int(args.max_items) > 0:
        n_cases = min(
            len(dataset.case_dirs),
            max(1, (int(args.max_items) + int(args.patches_per_image) - 1) // int(args.patches_per_image)),
        )
        selection_rng = random.Random(int(args.selection_seed))
        selected_indices = sorted(selection_rng.sample(range(len(dataset.case_dirs)), n_cases))
        dataset.case_dirs = [dataset.case_dirs[index] for index in selected_indices]
        print(
            f"selected {n_cases}/{total_cases} "
            f"cases with selection_seed={args.selection_seed}"
        )

    loader = DataLoader(
        dataset,
        batch_size=int(args.mbs),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    xs: List[torch.Tensor] = []
    ps: List[torch.Tensor] = []
    ms: List[torch.Tensor] = []
    samples_all: List[torch.Tensor] = []
    metas: List[Dict[str, Any]] = []

    remaining = int(args.max_items) if int(args.max_items) > 0 else None
    for batch in tqdm(loader, desc="generate invariance samples"):
        x = batch["x"].to(device).float()
        pred = batch["p"].to(device).float()
        mask = batch["m"].to(device).float()
        if remaining is not None:
            keep = min(int(remaining), int(x.shape[0]))
            x = x[:keep]
            pred = pred[:keep]
            mask = mask[:keep]
            remaining -= keep

        cond = torch.cat([pred, mask], dim=1)
        batch_samples = []
        for _ in range(int(args.num_samples)):
            v = torch.randn((x.shape[0], int(getattr(cinn, "module", cinn).flow.in_channels), 1, 1), device=device)
            z_sample = cinn.reverse(v, cond)
            x_sample = ae.decode(z_sample, im_cond=cond)
            batch_samples.append(x_sample.detach().cpu())

        xs.append(x.detach().cpu())
        ps.append(pred.detach().cpu())
        ms.append(mask.detach().cpu())
        samples_all.append(torch.stack(batch_samples, dim=1))

        batch_size = int(x.shape[0])
        for bi in range(batch_size):
            meta = {}
            for key, value in batch.items():
                if key in {"x", "p", "m"}:
                    continue
                try:
                    meta[key] = value[bi].item() if torch.is_tensor(value) else value[bi]
                except Exception:
                    pass
            metas.append(meta)

        if remaining is not None and remaining <= 0:
            break

    bank = {
        "x": torch.cat(xs, dim=0),
        "p": torch.cat(ps, dim=0),
        "m": torch.cat(ms, dim=0),
        "samples": torch.cat(samples_all, dim=0),
        "meta": metas,
        "config": vars(args),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(bank, args.out)
    print(f"saved sample bank: {args.out}")
    print(f"x: {tuple(bank['x'].shape)} samples: {tuple(bank['samples'].shape)}")


if __name__ == "__main__":
    main()
'''
generate dml val sample banks for dudonet

python -u -m MAR_Inv.evaluation.generate_samples \
  --vae_checkpoint "results/runs/cvae/cvae_dudonet/checkpoints/best.pt" \
  --cinn_checkpoint "results/runs/cinn/cinn_dudonet/cinn_best.pt" \
  --datafolder "/mnt/public/bahaa/dataset_validation" \
  --stats "configs/head_body_stats.json" \
  --fn_x "input_img.mha" \
  --fn_y "output_img.mha" \
  --fn_mask "input_mask.mha" \
  --fn_pred "pred_dudonet_img.mha" \
  --pred_name dudonet \
  --out results/sample_banks/dml_banks/dudonet_val_k2_new.pt \
  --num_samples 2 \
  --max_items 0 \
  --patches_per_image 1 \
  --patchsize 128 \
  --mbs 5 \
  --num_workers 0 \
  --strong_ratio 0.7 \
  --topk 5000 \
  --use_strength \
  --seed 1332 \
  --selection_seed 1332
'''