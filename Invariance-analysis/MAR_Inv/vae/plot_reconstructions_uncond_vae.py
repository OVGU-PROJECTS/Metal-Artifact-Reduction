"""Create qualitative reconstruction examples for a trained VAE.

This script is intended mainly for the unconditional VAE used as the embedding
model in the invariance evaluation. It loads a VAE checkpoint, reconstructs a
small number of validation patches, and saves one side-by-side image per patch.

It also supports the conditional VAE if the YAML has ``conditional: true`` and
if the dataset returns the conditioning tensors ``p`` and ``m``.
"""

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from data.patches_generator import PatchesGenerator
from MAR_Inv.vae.network import BigAE


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_generator_state(checkpoint_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("generator_state_dict", "model_state_dict", "state_dict"):
            if key in checkpoint:
                return strip_module_prefix(checkpoint[key])
    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return strip_module_prefix(checkpoint)
    raise KeyError(
        "Could not find a generator state dict. Expected one of: "
        "'generator_state_dict', 'model_state_dict', or 'state_dict'."
    )


def build_dataset(cfg: Dict, split: str, datafolder_override: str = "") -> PatchesGenerator:
    cfg = dict(cfg)
    if datafolder_override:
        datafolder = datafolder_override
    elif split == "train":
        datafolder = cfg.get("train_datafolder", cfg.get("datafolder", ""))
    else:
        datafolder = cfg.get("val_datafolder", cfg.get("datafolder", ""))
    if not datafolder:
        raise KeyError(f"No datafolder found for split '{split}'.")

    cfg["datafolder"] = datafolder
    args = argparse.Namespace(**cfg)

    return PatchesGenerator(
        args,
        fn_x=cfg.get("fn_x", "input_img.mha"),
        fn_y=cfg.get("fn_y", "output_img.mha"),
        fn_mask=cfg.get("fn_mask", "input_mask.mha"),
        patches_per_image=int(cfg.get("patches_per_image", 1)),
        strong_ratio=float(cfg.get("strong_ratio", 0.7)),
        topk=int(cfg.get("topk", 5000)),
        use_strength=bool(cfg.get("use_strength", True)),
        return_meta=True,
        stats_path=cfg["stats"],
        conditional=bool(cfg.get("conditional", False)),
        require_cond_files=bool(cfg.get("conditional", False)),
    )


def select_indices(length: int, n: int, seed: int, random_examples: bool) -> List[int]:
    n = min(int(n), int(length))
    if random_examples:
        rng = np.random.default_rng(int(seed))
        return rng.choice(length, size=n, replace=False).astype(int).tolist()
    return list(range(n))


def parse_indices(indices: str) -> Optional[List[int]]:
    if not indices.strip():
        return None
    parsed = [int(x.strip()) for x in indices.split(",") if x.strip()]
    if not parsed:
        raise ValueError("--indices was provided but no valid integer indices were parsed.")
    return parsed


def _base_dataset(dataset):
    return dataset.dataset if isinstance(dataset, Subset) else dataset


def _as_float_attr(obj, names: Sequence[str]) -> Optional[float]:
    for name in names:
        if hasattr(obj, name):
            return float(getattr(obj, name))
    return None


def extract_fov(batch: Dict) -> Optional[torch.Tensor]:
    """Return a 0=head, 1=body tensor when such metadata exists in the batch."""
    for key in ("fov", "fov_id", "fov_type", "is_body", "body"):
        if key not in batch:
            continue
        value = batch[key]
        if torch.is_tensor(value):
            return value.detach().cpu().view(-1)
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value).view(-1)
        if isinstance(value, (list, tuple)):
            mapped = []
            for item in value:
                if isinstance(item, str):
                    mapped.append(1 if item.lower().startswith("body") else 0)
                else:
                    mapped.append(int(item))
            return torch.as_tensor(mapped)
    return None


def denormalize_to_hu(
    x: torch.Tensor,
    fov: Optional[torch.Tensor],
    dataset: PatchesGenerator,
) -> torch.Tensor:
    """Convert z-scored patches back to HU using dataset statistics.

    If head/body statistics are available, fov must identify each patch with
    0=head and 1=body. If only global mean/std are available, those are used.
    """
    ds = _base_dataset(dataset)
    x_hu = x.detach().clone()

    global_mean = _as_float_attr(ds, ("mean", "mean_all", "global_mean"))
    global_std = _as_float_attr(ds, ("std", "std_all", "global_std"))
    head_mean = _as_float_attr(ds, ("mean_head", "head_mean"))
    head_std = _as_float_attr(ds, ("std_head", "head_std"))
    body_mean = _as_float_attr(ds, ("mean_body", "body_mean"))
    body_std = _as_float_attr(ds, ("std_body", "body_std"))

    if head_mean is not None and head_std is not None and body_mean is not None and body_std is not None:
        if fov is None:
            raise KeyError(
                "Cannot denormalize with separate head/body statistics because no FOV metadata was found "
                "in the batch. Disable --denorm_hu or make the dataset return a key such as 'fov'."
            )
        fov = fov.detach().cpu().view(-1)
        for i in range(x_hu.shape[0]):
            if int(fov[i].item()) == 0:
                mean, std = head_mean, head_std
            else:
                mean, std = body_mean, body_std
            x_hu[i] = x_hu[i] * float(std) + float(mean)
        return x_hu

    if global_mean is not None and global_std is not None:
        return x_hu * float(global_std) + float(global_mean)

    raise AttributeError(
        "Could not find normalization statistics on the dataset. Expected either "
        "mean/std, global_mean/global_std, or mean_head/std_head and mean_body/std_body."
    )


def get_conditioning(batch: Dict, conditional: bool, device: torch.device) -> Optional[torch.Tensor]:
    if not conditional:
        return None
    if "p" not in batch or "m" not in batch:
        raise KeyError("conditional=True requires batch keys 'p' and 'm'.")
    return torch.cat([batch["p"], batch["m"]], dim=1).to(device).float()


@torch.no_grad()
def reconstruct_batch(
    model: BigAE,
    x: torch.Tensor,
    use_posterior_mean: bool,
    im_cond: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if im_cond is not None:
        posterior = model.encode(torch.cat([x, im_cond], dim=1))
    else:
        posterior = model.encode(x)
    z = posterior.mode() if use_posterior_mean else posterior.sample()
    return model.decode(z, im_cond=im_cond)


def tensor_to_numpy_image(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().squeeze(0).numpy()


def robust_window(images: List[np.ndarray], low: float, high: float) -> Tuple[float, float]:
    pixels = np.concatenate([im.reshape(-1) for im in images])
    vmin, vmax = np.percentile(pixels, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.min(pixels)), float(np.max(pixels))
    return float(vmin), float(vmax)


def _example_output_path(output: str, row: int, dataset_index: int) -> str:
    output_abs = os.path.abspath(output)
    root, ext = os.path.splitext(output_abs)
    if ext:
        out_dir = os.path.dirname(output_abs)
        base = os.path.basename(root)
    else:
        out_dir = output_abs
        base = "vae_reconstruction"
        ext = ".png"

    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}_{row + 1:02d}_idx{int(dataset_index):06d}{ext}")


def save_figures(
    x: torch.Tensor,
    recon: torch.Tensor,
    output: str,
    indices: Sequence[int],
    #title: str,
    window_percentiles: Tuple[float, float],
    dpi: int,
) -> List[str]:
    n = x.shape[0]
    originals = [tensor_to_numpy_image(x[i]) for i in range(n)]
    recons = [tensor_to_numpy_image(recon[i]) for i in range(n)]
    vmin, vmax = robust_window(originals + recons, *window_percentiles)

    saved_paths: List[str] = []
    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.7), squeeze=False)
        axes = axes[0]
        axes[0].imshow(originals[i], cmap="gray", vmin=vmin, vmax=vmax)
        axes[0].set_title("Corrupted patch")
        axes[1].imshow(recons[i], cmap="gray", vmin=vmin, vmax=vmax)
        axes[1].set_title("VAE reconstruction")
        for ax in axes:
            ax.axis("off")

        #if title:
        #    fig.suptitle(f"{title} - example {i + 1}, dataset index {indices[i]}")
        fig.tight_layout()
        out_path = _example_output_path(output, i, int(indices[i]))
        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def save_npz(
    path: str,
    x: torch.Tensor,
    recon: torch.Tensor,
    indices: Sequence[int],
    denormalized: bool,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        x=x.detach().cpu().numpy(),
        recon=recon.detach().cpu().numpy(),
        indices=np.asarray(indices, dtype=np.int64),
        denormalized=np.asarray([bool(denormalized)]),
    )


def build_model(cfg: Dict, device: torch.device, strict_model_cfg: bool) -> BigAE:
    required = [
        "patchsize",
        "z_dim",
        "encoder_norm",
        "encoder_type",
        "encoder_pretrained",
        "decoder_chn",
        "decoder_norm",
    ]
    missing = [key for key in required if key not in cfg]
    if missing and strict_model_cfg:
        raise KeyError(f"Missing required model config keys: {missing}")

    # Safe defaults only used when --non_strict_model_cfg is passed.
    defaults = {
        "patchsize": 128,
        "z_dim": 256,
        "encoder_norm": "bn",
        "encoder_type": "resnet50",
        "encoder_pretrained": False,
        "decoder_chn": 64,
        "decoder_norm": "bn",
    }
    model_cfg = dict(defaults)
    model_cfg.update(cfg)

    conditional = bool(model_cfg.get("conditional", False))
    cond_ch = 2 if conditional else 0
    args = argparse.Namespace(**model_cfg)
    return BigAE(args, in_ch=1 + cond_ch, out_ch=1, cond_ch=cond_ch).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct validation patches with a trained VAE checkpoint and save a qualitative figure."
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config used for the VAE run.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt, last.pt, or another VAE checkpoint.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Output prefix/path. If a file path is given, one image per example is saved "
            "with a numbered suffix before the extension."
        ),
    )
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"), help="Dataset split to sample from.")
    parser.add_argument("--datafolder", type=str, default="", help="Optional override for the split data folder.")
    parser.add_argument("--n", type=int, default=10, help="Number of patches to reconstruct.")
    parser.add_argument("--indices", type=str, default="", help="Comma-separated dataset indices. Overrides --n/--random_examples.")
    parser.add_argument("--random_examples", action="store_true", help="Sample random examples instead of the first n.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random example selection and posterior sampling.")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size. Default: use n in one batch.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for this plotting script.")
    parser.add_argument("--posterior_sample", action="store_true", help="Sample from q(z|x) instead of using the posterior mean.")
    parser.add_argument("--denorm_hu", action="store_true", help="Denormalize patches back to HU before plotting and saving.")
    parser.add_argument("--window_low", type=float, default=1.0, help="Lower percentile for display window.")
    parser.add_argument("--window_high", type=float, default=99.0, help="Upper percentile for display window.")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    parser.add_argument("--title", type=str, default="", help="Optional figure title.")
    parser.add_argument("--save_npz", type=str, default="", help="Optional .npz path for reconstructed tensors.")
    parser.add_argument("--device", type=str, default="", help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--allow_pretrained_init",
        action="store_true",
        help="Allow torchvision to initialize the encoder with pretrained weights before loading the checkpoint. "
        "By default this is disabled to avoid unnecessary downloads.",
    )
    parser.add_argument(
        "--non_strict_state",
        action="store_true",
        help="Load the checkpoint with strict=False and print missing/unexpected keys.",
    )
    parser.add_argument(
        "--non_strict_model_cfg",
        action="store_true",
        help="Use fallback model defaults if some architecture keys are missing from the YAML.",
    )
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    cfg = load_yaml(args.config)

    # The checkpoint fully defines the trained weights; pretrained initialization is unnecessary for plotting.
    if not args.allow_pretrained_init:
        cfg["encoder_pretrained"] = False

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = build_dataset(cfg, split=args.split, datafolder_override=args.datafolder)
    explicit_indices = parse_indices(args.indices)
    if explicit_indices is None:
        indices = select_indices(len(dataset), args.n, args.seed, args.random_examples)
    else:
        bad = [idx for idx in explicit_indices if idx < 0 or idx >= len(dataset)]
        if bad:
            raise IndexError(f"Dataset indices out of range: {bad}. Dataset length is {len(dataset)}.")
        indices = explicit_indices

    subset = Subset(dataset, indices)
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else len(indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg, device=device, strict_model_cfg=not args.non_strict_model_cfg)
    state = load_generator_state(args.checkpoint, device=device)
    load_msg = model.load_state_dict(state, strict=not args.non_strict_state)
    if args.non_strict_state:
        print("[load_state_dict]", load_msg)
    model.eval()

    x_batches: List[torch.Tensor] = []
    recon_batches: List[torch.Tensor] = []
    fov_batches: List[torch.Tensor] = []

    conditional = bool(cfg.get("conditional", False))
    use_posterior_mean = not bool(args.posterior_sample)

    for batch in loader:
        if "x" not in batch:
            raise KeyError(f"Expected dataset batch to contain key 'x'. Available keys: {list(batch.keys())}")
        x = batch["x"].to(device).float()
        im_cond = get_conditioning(batch, conditional=conditional, device=device)
        recon = reconstruct_batch(model, x, use_posterior_mean=use_posterior_mean, im_cond=im_cond)
        x_batches.append(x.detach().cpu())
        recon_batches.append(recon.detach().cpu())
        fov = extract_fov(batch)
        if fov is not None:
            fov_batches.append(fov.detach().cpu())

    x_all = torch.cat(x_batches, dim=0)
    recon_all = torch.cat(recon_batches, dim=0)
    fov_all = torch.cat(fov_batches, dim=0) if fov_batches else None

    if args.denorm_hu:
        x_plot = denormalize_to_hu(x_all, fov_all, dataset)
        recon_plot = denormalize_to_hu(recon_all, fov_all, dataset)
    else:
        x_plot = x_all
        recon_plot = recon_all


    saved_paths = save_figures(
        x=x_plot,
        recon=recon_plot,
        output=args.output,
        indices=indices,
        window_percentiles=(float(args.window_low), float(args.window_high)),
        dpi=int(args.dpi),
    )

    if args.save_npz:
        save_npz(args.save_npz, x_plot, recon_plot, indices=indices, denormalized=bool(args.denorm_hu))

    print("Saved figures:")
    for path in saved_paths:
        print(f"  {path}")
    print(f"Dataset split: {args.split}; selected indices: {indices}")
    if args.save_npz:
        print(f"Saved arrays: {args.save_npz}")


if __name__ == "__main__":
    main()
'''
 Example: generate posterior-mean reconstructions with the unconditional VAE

 Run from the Invariance-analysis repository root:

   cd /mnt/public/bahaa/Invariance-analysis

   OUTPUT_DIR="results/posterior_mean_reconstructions/test"
   mkdir -p "$OUTPUT_DIR"

   python -u -m MAR_Inv.vae.plot_reconstructions_uncond_vae \
     --config configs/uncond.yaml \
     --checkpoint results/runs/evaluation/uncond_vae/checkpoints/best.pt \
     --output "$OUTPUT_DIR/uncond_vae_reconstruction.png" \
     --split val \
     --indices 1662,1956,1119 \
     --batch_size 2 \
     --num_workers 0 \
     --device cuda:0 \
     --denorm_hu \
     --window_low 1 \
     --window_high 99 \
     --dpi 250 \
     --save_npz "$OUTPUT_DIR/uncond_vae_reconstructions.npz"

 To select random validation patches instead, replace --indices with:

   --n 10 \
   --random_examples \
   --seed 1332
'''