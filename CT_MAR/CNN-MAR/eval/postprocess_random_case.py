#!/usr/bin/env python3
"""Post-process one randomly selected validation case.

The script uses an already saved ``pred_cnnmar_img.mha``.  It does not load or
run the CNN.  By default, results are written under
``results/random_postprocessing_example/<case_id>/``.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mar_common.cases import fov_from_folder, list_case_dirs
from mar_common.scanner_config import recon_config
from geometry.fanbeam import RECO_FOV
from mar_common.io import read_mha, write_mha
from postprocessing import postprocess_cnnmar


CNNMAR_DIR = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "input_proj.mha",
    "input_mask.mha",
    "input_img.mha",
    "img_li.mha",
    "img_bhc.mha",
    "pred_cnnmar_img.mha",
    "output_img.mha",
)


def _scanner_fov_mm(case: Path) -> float:
    value = fov_from_folder(str(case))
    if value in ("head", 0, "0"):
        return float(RECO_FOV["head"])
    if value in ("body", 1, "1"):
        return float(RECO_FOV["body"])
    raise ValueError(f"unsupported FOV value {value!r} for case {case.name}")


def _eligible_cases(data_dir: Path) -> list[Path]:
    return [
        case
        for case in list_case_dirs(data_dir)
        if all((case / filename).is_file() for filename in REQUIRED_FILES)
    ]


def _reinsert_metal(
    image_hu: np.ndarray,
    input_hu: np.ndarray,
    original_metal_mask: np.ndarray,
) -> np.ndarray:
    """Copy measured input values into the original metal region for display."""
    image = np.asarray(image_hu)
    input_image = np.asarray(input_hu)
    metal = np.asarray(original_metal_mask) > 0
    if image.shape != input_image.shape or metal.shape != image.shape:
        raise ValueError(
            "image, input image, and original metal mask must have equal shapes; "
            f"got {image.shape}, {input_image.shape}, and {metal.shape}"
        )
    displayed = image.copy()
    displayed[metal] = input_image[metal]
    return displayed


def _save_comparison(
    output_path: Path,
    input_hu: np.ndarray,
    li_hu: np.ndarray,
    bhc_hu: np.ndarray,
    cnn_hu: np.ndarray,
    prior_hu: np.ndarray,
    final_hu: np.ndarray,
    reference_hu: np.ndarray,
    original_metal_mask: np.ndarray,
    *,
    window_min: float,
    window_max: float,
) -> None:
    li_display = _reinsert_metal(li_hu, input_hu, original_metal_mask)
    bhc_display = _reinsert_metal(bhc_hu, input_hu, original_metal_mask)
    cnn_display = _reinsert_metal(cnn_hu, input_hu, original_metal_mask)
    final_display = _reinsert_metal(final_hu, input_hu, original_metal_mask)
    reference_display = _reinsert_metal(
        reference_hu, input_hu, original_metal_mask
    )

    top_panels = [
        ("Input", input_hu),
        ("LI", li_display),
        ("BHC", bhc_display),
        ("Direct CNN", cnn_display),
    ]
    bottom_panels = [
        ("CNN prior", prior_hu),
        ("Final CNN-MAR", final_display),
        ("Reference", reference_display),
    ]

    figure = plt.figure(figsize=(14.0, 8.5))
    grid = figure.add_gridspec(2, 8, hspace=0.12, wspace=0.10)
    axes = [
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
        figure.add_subplot(grid[0, 6:8]),
        figure.add_subplot(grid[1, 1:3]),
        figure.add_subplot(grid[1, 3:5]),
        figure.add_subplot(grid[1, 5:7]),
    ]
    for axis, (title, image) in zip(axes, top_panels + bottom_panels):
        axis.imshow(image, cmap="gray", vmin=window_min, vmax=window_max)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly select one validation case with an existing CNN-MAR "
            "prediction and save its post-processing stages."
        )
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        type=Path,
        help="Validation dataset containing case subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=CNNMAR_DIR / "results" / "random_postprocessing_example",        help="Destination root (default: results/random_postprocessing_example).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible case selection.",
    )
    parser.add_argument("--window-min", type=float, default=-1000.0)
    parser.add_argument("--window-max", type=float, default=2000.0)
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")
    if args.window_min >= args.window_max:
        raise ValueError("--window-min must be smaller than --window-max")

    eligible = _eligible_cases(args.data_dir)
    if not eligible:
        required = ", ".join(REQUIRED_FILES)
        raise RuntimeError(
            f"no eligible cases found in {args.data_dir}; required: {required}"
        )

    selected = random.Random(args.seed).choice(eligible)
    destination = args.output_dir / selected.name
    destination.mkdir(parents=True, exist_ok=True)

    measured_projection = read_mha(selected / "input_proj.mha")
    metal_mask = read_mha(selected / "input_mask.mha")
    input_hu = read_mha(selected / "input_img.mha")
    li_hu = read_mha(selected / "img_li.mha")
    bhc_hu = read_mha(selected / "img_bhc.mha")
    cnn_hu = read_mha(selected / "pred_cnnmar_img.mha")
    reference_hu = read_mha(selected / "output_img.mha")
    trace_path = selected / "input_metal_trace_proj.mha"
    metal_trace = read_mha(trace_path) if trace_path.is_file() else None

    config = recon_config(_scanner_fov_mm(selected))
    result = postprocess_cnnmar(
        measured_projection,
        cnn_hu,
        metal_mask,
        config,
        metal_trace=metal_trace,
    )

    write_mha(input_hu, destination / "input_img.mha")
    write_mha(li_hu, destination / "img_li.mha")
    write_mha(bhc_hu, destination / "img_bhc.mha")
    write_mha(cnn_hu, destination / "pred_cnnmar_img.mha")
    write_mha(result.prior_hu, destination / "pred_cnnmar_prior.mha")
    write_mha(result.image_hu, destination / "pred_cnnmar_final.mha")
    write_mha(result.corrected_projection, destination / "corrected_projection.mha")
    write_mha(metal_mask, destination / "metal_mask.mha")
    write_mha(reference_hu, destination / "output_img.mha")

    write_mha(
        _reinsert_metal(li_hu, input_hu, metal_mask),
        destination / "img_li_with_metal.mha",
    )
    write_mha(
        _reinsert_metal(bhc_hu, input_hu, metal_mask),
        destination / "img_bhc_with_metal.mha",
    )
    write_mha(
        _reinsert_metal(cnn_hu, input_hu, metal_mask),
        destination / "pred_cnnmar_img_with_metal.mha",
    )
    write_mha(
        _reinsert_metal(result.image_hu, input_hu, metal_mask),
        destination / "pred_cnnmar_final_with_metal.mha",
    )
    write_mha(
        _reinsert_metal(reference_hu, input_hu, metal_mask),
        destination / "output_img_with_metal.mha",
    )

    _save_comparison(
        destination / "comparison.png",
        input_hu,
        li_hu,
        bhc_hu,
        cnn_hu,
        result.prior_hu,
        result.image_hu,
        reference_hu,
        metal_mask,
        window_min=args.window_min,
        window_max=args.window_max,
    )

    metadata = {
        "selected_case": selected.name,
        "source_case": str(selected),
        "seed": args.seed,
        "fov_mm": config.recon.fov,
        "cluster_centers_hu": result.prior_info.cluster_centers_hu,
        "air_threshold_hu": result.prior_info.air_threshold_hu,
        "bone_threshold_hu": result.prior_info.bone_threshold_hu,
        "water_mean_hu": result.prior_info.water_mean_hu,
        "used_existing_metal_trace": trace_path.is_file(),
        "comparison_uses_input_metal_values": True,
        "display_window_hu": [args.window_min, args.window_max],
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"[selected] {selected}")
    print(f"[written]  {destination}")
    print(
        "[outputs] input_img.mha, img_li.mha, img_bhc.mha, "
        "pred_cnnmar_img.mha, pred_cnnmar_prior.mha, "
        "pred_cnnmar_final.mha, output_img.mha, comparison.png"
    )


if __name__ == "__main__":
    main()
'''
conda activate myenv
cd /mnt/public/bahaa/compare/CT_MAR/CNN-MAR

python -u -m eval.postprocess_random_case \
  --data_dir /mnt/public/bahaa/dataset_validation \
  --output_dir results/random_postprocessing_examples \
  --seed 42
  '''