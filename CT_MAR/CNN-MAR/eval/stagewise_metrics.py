#!/usr/bin/env python3
"""Compute stage-wise CNN-MAR RMSE, training SSIM, and MAE in HU.

Each case directory must contain the metal-free reference, metal mask, and all
five pipeline images listed in ``STAGES`` below. SSIM calls
the exact ``ssim_aapm`` function used by the CNN-MAR training callback.

The script writes:

* ``stagewise_per_case.csv``: one row per case and pipeline stage;
* ``stagewise_summary.csv``: mean and sample standard deviation per stage;
* ``stagewise_latex_rows.tex``: table rows.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


STAGES = (
    ("raw", "Uncorrected, $x_{\\mathrm{raw}}$", "input_img.mha"),
    ("li", "Linear interpolation, $x_{\\mathrm{LI}}$", "img_li.mha"),
    ("bhc", "Beam-hardening correction, $x_{\\mathrm{BHC}}$", "img_bhc.mha"),
    ("cnn", "Network output, $x_{\\mathrm{CNN}}$", "pred_cnnmar_img.mha"),
    (
        "final",
        "Final CNN-MAR, $x_{\\mathrm{CNN\\text{-}MAR}}$",
        "pred_cnnmar_final.mha",
    ),
)


@dataclass(frozen=True)
class Metrics:
    rmse_hu: float
    ssim_aapm_hu: float
    mae_hu: float


def _image2d(path: Path) -> np.ndarray:
    from mar_common.io import read_mha

    image = np.asarray(read_mha(path), dtype=np.float32).squeeze()
    if image.ndim != 2:
        raise ValueError(f"{path} has shape {image.shape}; expected a 2-D image")
    if not np.isfinite(image).all():
        raise ValueError(f"{path} contains NaN or infinite values")
    return image


def aapm_evaluation_mask(
    reference_hu: np.ndarray,
    dilated_metal_mask: np.ndarray,
) -> np.ndarray:
    """Return the patient/metal mask used by training's AAPM SSIM."""
    mask = (
        np.asarray(reference_hu) > -950.0
    ) & ~(np.asarray(dilated_metal_mask).squeeze() > 0)
    if not np.any(mask):
        raise ValueError("AAPM evaluation mask is empty")
    return mask


def compute_metrics(
    reference_hu: np.ndarray,
    estimate_hu: np.ndarray,
    dilated_metal_mask: np.ndarray,
    *,
    hu_min: float,
    hu_max: float,
) -> Metrics:
    """Compute errors and the exact AAPM SSIM used during training."""
    from metrics.numpy_ssim import ssim_aapm

    mask = aapm_evaluation_mask(reference_hu, dilated_metal_mask)
    reference = np.clip(reference_hu, hu_min, hu_max)
    estimate = np.clip(estimate_hu, hu_min, hu_max)
    difference = estimate.astype(np.float64) - reference.astype(np.float64)
    mae = float(np.mean(np.abs(difference[mask]), dtype=np.float64))
    mse = float(np.mean(np.square(difference[mask]), dtype=np.float64))
    return Metrics(
        rmse_hu=float(np.sqrt(mse)),
        ssim_aapm_hu=ssim_aapm(
            reference_hu,
            estimate_hu,
            dilated_metal_mask,
        ),
        mae_hu=mae,
    )


def _mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    # The table describes variation over test cases, hence sample SD (ddof=1).
    sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return float(np.mean(array)), sd


def _write_outputs(
    output_dir: Path,
    rows: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_case_path = output_dir / "stagewise_per_case.csv"
    fields = (
        "case",
        "stage",
        "rmse_hu",
        "ssim_aapm_hu",
        "mae_hu",
        "pixels",
    )
    with per_case_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, object]] = []
    latex_rows: list[str] = []
    for stage_key, latex_label, _ in STAGES:
        selected = [row for row in rows if row["stage"] == stage_key]
        rmse_mean, rmse_sd = _mean_sd([float(row["rmse_hu"]) for row in selected])
        ssim_mean, ssim_sd = _mean_sd(
            [float(row["ssim_aapm_hu"]) for row in selected]
        )
        mae_mean, mae_sd = _mean_sd([float(row["mae_hu"]) for row in selected])
        summary_rows.append(
            {
                "stage": stage_key,
                "cases": len(selected),
                "rmse_mean_hu": rmse_mean,
                "rmse_sd_hu": rmse_sd,
                "ssim_aapm_hu_mean": ssim_mean,
                "ssim_aapm_hu_sd": ssim_sd,
                "mae_mean_hu": mae_mean,
                "mae_sd_hu": mae_sd,
            }
        )
        latex_rows.append(
            f"{latex_label} & {rmse_mean:.2f} $\\pm$ {rmse_sd:.2f} "
            f"& {ssim_mean:.4f} $\\pm$ {ssim_sd:.4f} "
            f"& {mae_mean:.2f} $\\pm$ {mae_sd:.2f} \\\\"
        )

    summary_path = output_dir / "stagewise_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    latex_path = output_dir / "stagewise_latex_rows.tex"
    latex_path.write_text("\n".join(latex_rows) + "\n", encoding="utf-8")

    print(f"[written] {per_case_path}")
    print(f"[written] {summary_path}")
    print(f"[written] {latex_path}")
    print("\nLaTeX rows (RMSE, AAPM SSIM, MAE):")
    print("\n".join(latex_rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all CNN-MAR stages against the metal-free reference."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        type=Path,
        help="Evaluation-set directory containing one subdirectory per case.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("stagewise_metrics"),
        help="Output directory (default: ./stagewise_metrics).",
    )
    parser.add_argument("--reference-name", default="output_img.mha")
    parser.add_argument("--mask-name", default="input_mask_dilated.mha")
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=3000.0)
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        parser.error(f"data directory not found: {args.data_dir}")
    if args.hu_max <= args.hu_min:
        parser.error("--hu-max must be greater than --hu-min")

    required_names = (
        args.reference_name,
        args.mask_name,
        *(filename for _, _, filename in STAGES),
    )
    complete_cases: list[Path] = []
    for case in sorted(path for path in args.data_dir.iterdir() if path.is_dir()):
        missing = [name for name in required_names if not (case / name).is_file()]
        if missing:
            print(f"[skip] {case.name}: missing {', '.join(missing)}")
        else:
            complete_cases.append(case)
    if not complete_cases:
        raise RuntimeError(
            "no complete evaluation cases were found; generate "
            "pred_cnnmar_final.mha first with: python -m eval.postprocess "
            "--data_dir <evaluation-set>"
        )

    rows: list[dict[str, object]] = []
    for index, case in enumerate(complete_cases, start=1):
        reference = _image2d(case / args.reference_name)
        dilated_metal_mask = _image2d(case / args.mask_name)
        mask = aapm_evaluation_mask(reference, dilated_metal_mask)
        for stage_key, _, filename in STAGES:
            estimate = _image2d(case / filename)
            if estimate.shape != reference.shape:
                raise ValueError(
                    f"{case.name}/{filename} shape {estimate.shape} does not "
                    f"match reference shape {reference.shape}"
                )
            metrics = compute_metrics(
                reference,
                estimate,
                dilated_metal_mask,
                hu_min=args.hu_min,
                hu_max=args.hu_max,
            )
            rows.append(
                {
                    "case": case.name,
                    "stage": stage_key,
                    "rmse_hu": f"{metrics.rmse_hu:.10g}",
                    "ssim_aapm_hu": f"{metrics.ssim_aapm_hu:.10g}",
                    "mae_hu": f"{metrics.mae_hu:.10g}",
                    "pixels": int(mask.sum()),
                }
            )
        print(f"[{index}/{len(complete_cases)}] {case.name}")

    _write_outputs(args.out_dir, rows)
    print(
        f"[done] evaluated {len(complete_cases)} cases; "
        f"HU range=[{args.hu_min:g}, {args.hu_max:g}]"
    )


if __name__ == "__main__":
    main()
'''
conda activate myenv
cd /mnt/public/bahaa/compare/CT_MAR/CNN-MAR

python -u -m eval.stagewise_metrics \
  --data_dir /mnt/public/bahaa/dataset_validation \
  --out_dir results/stagewise_metrics_validation 
  '''