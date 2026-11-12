"""Create qualitative reconstruction examples for a conditional VAE.

For every selected patch, the script saves a three-panel comparison containing
the corrupted input, the MAR prediction used in the condition, and the cVAE
reconstruction. """

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from MAR_Inv.vae.plot_reconstructions_uncond_vae import (
    build_dataset,
    build_model,
    denormalize_to_hu,
    extract_fov,
    get_conditioning,
    load_generator_state,
    load_yaml,
    parse_indices,
    reconstruct_batch,
    robust_window,
    select_indices,
    tensor_to_numpy_image,
)


def method_display_name(cfg: Dict) -> str:
    value = str(cfg.get("pred_name") or cfg.get("greybox") or "MAR")
    names = {
        "cnnmar": "CNN-MAR",
        "dudonet": "DuDoNet",
        "pconv": "PConv-MAR",
    }
    return names.get(value.lower(), value)


def example_output_path(output: str, row: int, dataset_index: int) -> str:
    output_abs = os.path.abspath(output)
    root, ext = os.path.splitext(output_abs)
    if ext:
        output_dir = os.path.dirname(output_abs)
        base = os.path.basename(root)
    else:
        output_dir = output_abs
        base = "cvae_reconstruction"
        ext = ".png"

    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(
        output_dir,
        f"{base}_{row + 1:02d}_idx{int(dataset_index):06d}{ext}",
    )


def save_comparisons(
    corrupted: torch.Tensor,
    prediction: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    output: str,
    indices: Sequence[int],
    method_name: str,
    title: str,
    window_percentiles: Tuple[float, float],
    fixed_window_hu: Optional[Tuple[float, float]],
    dpi: int,
) -> List[str]:
    corrupted_images = [
        tensor_to_numpy_image(corrupted[i]) for i in range(corrupted.shape[0])
    ]
    prediction_images = [
        tensor_to_numpy_image(prediction[i]) for i in range(prediction.shape[0])
    ]
    reconstruction_images = [
        tensor_to_numpy_image(reconstruction[i])
        for i in range(reconstruction.shape[0])
    ]
    if fixed_window_hu is None:
        vmin, vmax = robust_window(
            corrupted_images + prediction_images + reconstruction_images,
            *window_percentiles,
        )
    else:
        vmin, vmax = fixed_window_hu

    saved_paths: List[str] = []
    for row, dataset_index in enumerate(indices):
        figure, axes = plt.subplots(1, 3, figsize=(8.1, 2.7), squeeze=False)
        axes = axes[0]
        panels = [
            ("Corrupted patch", corrupted_images[row]),
            (f"{method_name} condition", prediction_images[row]),
            ("cVAE reconstruction", reconstruction_images[row]),
        ]
        for axis, (panel_title, image) in zip(axes, panels):
            axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
            axis.set_title(panel_title)
            axis.axis("off")

        if title:
            figure.suptitle(
                f"{title} — example {row + 1}, dataset index {dataset_index}"
            )
        figure.tight_layout()
        path = example_output_path(output, row, int(dataset_index))
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(path)

    return saved_paths


def save_arrays(
    path: str,
    *,
    corrupted: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
    reconstruction: torch.Tensor,
    indices: Sequence[int],
    fov: Optional[torch.Tensor],
    denormalized: bool,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {
        "x": corrupted.numpy(),
        "p": prediction.numpy(),
        "m": mask.numpy(),
        "reconstruction": reconstruction.numpy(),
        "indices": np.asarray(indices, dtype=np.int64),
        "denormalized": np.asarray([bool(denormalized)]),
    }
    if fov is not None:
        arrays["fov"] = fov.numpy()
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct patches with a conditional VAE and save corrupted, "
            "condition-prediction, and reconstruction panels."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--datafolder", default="")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument(
        "--indices",
        default="",
        help="Comma-separated dataset indices; overrides --n and --random_examples.",
    )
    parser.add_argument("--random_examples", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--posterior_sample",
        action="store_true",
        help="Sample from q(z|x,c); the default uses the posterior mean.",
    )
    parser.add_argument("--denorm_hu", action="store_true")
    parser.add_argument("--window_low", type=float, default=1.0)
    parser.add_argument("--window_high", type=float, default=99.0)
    parser.add_argument(
        "--window_min_hu",
        type=float,
        default=None,
        help=(
            "Optional fixed lower HU display limit. Must be used together with "
            "--window_max_hu and --denorm_hu. Overrides percentile windowing."
        ),
    )
    parser.add_argument(
        "--window_max_hu",
        type=float,
        default=None,
        help=(
            "Optional fixed upper HU display limit. Must be used together with "
            "--window_min_hu and --denorm_hu. Overrides percentile windowing."
        ),
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--title", default="")
    parser.add_argument("--save_npz", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--allow_pretrained_init", action="store_true")
    parser.add_argument("--non_strict_state", action="store_true")
    parser.add_argument("--non_strict_model_cfg", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.window_low < args.window_high <= 100.0:
        raise ValueError(
            "--window_low and --window_high must satisfy "
            "0 <= low < high <= 100"
        )
    fixed_window_requested = (
        args.window_min_hu is not None or args.window_max_hu is not None
    )
    if fixed_window_requested:
        if args.window_min_hu is None or args.window_max_hu is None:
            raise ValueError(
                "--window_min_hu and --window_max_hu must be provided together"
            )
        if not args.denorm_hu:
            raise ValueError(
                "fixed HU windowing requires --denorm_hu"
            )
        if args.window_min_hu >= args.window_max_hu:
            raise ValueError(
                "--window_min_hu must be smaller than --window_max_hu"
            )

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    cfg = load_yaml(args.config)
    if not bool(cfg.get("conditional", False)):
        raise ValueError(
            "This script requires a conditional VAE configuration with "
            "'conditional: true'."
        )
    if not args.allow_pretrained_init:
        cfg["encoder_pretrained"] = False

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset = build_dataset(
        cfg,
        split=args.split,
        datafolder_override=args.datafolder,
    )

    explicit_indices = parse_indices(args.indices)
    if explicit_indices is None:
        indices = select_indices(
            len(dataset),
            args.n,
            args.seed,
            args.random_examples,
        )
    else:
        invalid = [
            index
            for index in explicit_indices
            if index < 0 or index >= len(dataset)
        ]
        if invalid:
            raise IndexError(
                f"Dataset indices out of range: {invalid}. "
                f"Dataset length is {len(dataset)}."
            )
        indices = explicit_indices
    if not indices:
        raise ValueError("No examples were selected.")

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=(
            int(args.batch_size)
            if int(args.batch_size) > 0
            else len(indices)
        ),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        cfg,
        device=device,
        strict_model_cfg=not args.non_strict_model_cfg,
    )
    state = load_generator_state(args.checkpoint, device=device)
    load_result = model.load_state_dict(
        state,
        strict=not args.non_strict_state,
    )
    if args.non_strict_state:
        print("[load_state_dict]", load_result)
    model.eval()

    corrupted_batches: List[torch.Tensor] = []
    prediction_batches: List[torch.Tensor] = []
    mask_batches: List[torch.Tensor] = []
    reconstruction_batches: List[torch.Tensor] = []
    fov_batches: List[torch.Tensor] = []

    for batch in loader:
        required = [key for key in ("x", "p", "m") if key not in batch]
        if required:
            raise KeyError(
                f"Conditional batch is missing {required}; "
                f"available keys: {list(batch.keys())}"
            )

        corrupted = batch["x"].to(device).float()
        condition = get_conditioning(
            batch,
            conditional=True,
            device=device,
        )
        reconstruction = reconstruct_batch(
            model,
            corrupted,
            use_posterior_mean=not args.posterior_sample,
            im_cond=condition,
        )

        corrupted_batches.append(corrupted.detach().cpu())
        prediction_batches.append(batch["p"].detach().cpu().float())
        mask_batches.append(batch["m"].detach().cpu().float())
        reconstruction_batches.append(reconstruction.detach().cpu())
        fov = extract_fov(batch)
        if fov is not None:
            fov_batches.append(fov.detach().cpu())

    corrupted_all = torch.cat(corrupted_batches, dim=0)
    prediction_all = torch.cat(prediction_batches, dim=0)
    mask_all = torch.cat(mask_batches, dim=0)
    reconstruction_all = torch.cat(reconstruction_batches, dim=0)
    fov_all = torch.cat(fov_batches, dim=0) if fov_batches else None

    if args.denorm_hu:
        corrupted_plot = denormalize_to_hu(
            corrupted_all,
            fov_all,
            dataset,
        )
        prediction_plot = denormalize_to_hu(
            prediction_all,
            fov_all,
            dataset,
        )
        reconstruction_plot = denormalize_to_hu(
            reconstruction_all,
            fov_all,
            dataset,
        )
    else:
        corrupted_plot = corrupted_all
        prediction_plot = prediction_all
        reconstruction_plot = reconstruction_all

    saved_paths = save_comparisons(
        corrupted_plot,
        prediction_plot,
        reconstruction_plot,
        output=args.output,
        indices=indices,
        method_name=method_display_name(cfg),
        title=args.title,
        window_percentiles=(args.window_low, args.window_high),
        fixed_window_hu=(
            (float(args.window_min_hu), float(args.window_max_hu))
            if fixed_window_requested
            else None
        ),
        dpi=args.dpi,
    )

    if args.save_npz:
        save_arrays(
            args.save_npz,
            corrupted=corrupted_plot,
            prediction=prediction_plot,
            mask=mask_all,
            reconstruction=reconstruction_plot,
            indices=indices,
            fov=fov_all,
            denormalized=args.denorm_hu,
        )

    print("Saved figures:")
    for path in saved_paths:
        print(f"  {path}")
    print(f"Dataset split: {args.split}; selected indices: {indices}")
    print(
        "Latent reconstruction: "
        + ("posterior sample" if args.posterior_sample else "posterior mean")
    )
    if args.save_npz:
        print(f"Saved arrays: {args.save_npz}")


if __name__ == "__main__":
    main()
'''
 Example: generate posterior-mean reconstructions for selected validation patches


   python -u -m MAR_Inv.vae.plot_reconstruction_cvae \
     --config "configs/dudonet.yaml" \
     --checkpoint "results/runs/cvae/cvae_dudonet/checkpoints/best.pt" \
     --output "results/posterior_mean_reconstructions/test/dudonet_cvae_reconstruction.png" \
     --split val \
     --indices 1662,1956,1119 \
     --batch_size 2 \
     --num_workers 0 \
     --device cuda:0 \
     --denorm_hu \
     --window_min_hu -1000 \
     --window_max_hu 2000 \
     --dpi 250 
'''