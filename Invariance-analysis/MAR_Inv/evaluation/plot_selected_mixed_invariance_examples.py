#!/usr/bin/env python3
"""Create a thesis figure from selected mixed head/body sample-bank examples.

The selected examples do not all come from the same subset or sample seed.

    Subset  Example Id  Sample seed
1   Body	717	        1332
2	Body	589     	1332
3	Body	666     	2026
4	Head	65      	26

The output figure has five rows:

  input patch
  artifact-free reference with metal pixels reinserted from the input
  PConv-MAR sample
  CNN-MAR sample
  DuDoNet sample

and one column per selected example.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover
    raise ImportError("SimpleITK is required to read output_img.mha.") from exc


METHODS = ("pconv", "cnnmar", "dudonet")


def load_bank(path: str | Path) -> Dict[str, Any]:
    bank = torch.load(path, map_location="cpu", weights_only=False)
    required = {"x", "p", "m", "samples", "meta"}
    missing = sorted(required - set(bank))
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")
    return bank


def read_stats(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_std(stats: Dict[str, Any], fov: int) -> Tuple[float, float]:
    key = "head" if int(fov) == 0 else "body"
    return float(stats[key]["mean"]), float(stats[key]["std"])


def denorm(tensor: torch.Tensor, mean: float, std: float) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    arr = np.squeeze(arr).astype(np.float32, copy=False)
    return arr * float(std) + float(mean)


def read_mha_2d(path: str | Path) -> np.ndarray:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return np.squeeze(arr).astype(np.float32, copy=False)


def scalar(value: Any) -> Any:
    return value.item() if torch.is_tensor(value) else value


def meta_value(meta: Dict[str, Any], key: str, default: Any = None) -> Any:
    return scalar(meta.get(key, default))


def check_alignment(subset: str, patch_index: int, banks: Dict[str, Dict[str, Any]]) -> None:
    anchors = {}
    for method, bank in banks.items():
        meta = bank["meta"][patch_index]
        anchors[method] = (
            str(meta_value(meta, "case_dir", "")),
            int(meta_value(meta, "cy", -1)),
            int(meta_value(meta, "cx", -1)),
            int(meta_value(meta, "patch_j", -1)),
        )
    if len(set(anchors.values())) != 1:
        raise ValueError(
            f"{subset} banks are not aligned at patch index {patch_index}: {anchors}"
        )


def sample_index(k: int, sample_seed: int, patch_index: int, method_offset: int) -> int:
    rng = random.Random((int(sample_seed) * 1_000_003) + int(patch_index) + method_offset * 10_000)
    return int(rng.randrange(int(k)))


def parse_example(text: str) -> Tuple[str, int, int]:
    """Parse subset:index:sample_seed, e.g. body:717:1332."""
    parts = [part.strip() for part in text.split(":")]
    if len(parts) != 3:
        raise ValueError(
            f"Example {text!r} must have format subset:index:sample_seed."
        )
    subset = parts[0].lower()
    if subset not in {"head", "body"}:
        raise ValueError(f"Subset must be head or body, got {subset!r}.")
    return subset, int(parts[1]), int(parts[2])


def reference_patch(meta: Dict[str, Any], patchsize: int, fn_y: str) -> np.ndarray:
    case_dir = Path(str(meta_value(meta, "case_dir")))
    cy = int(meta_value(meta, "cy"))
    cx = int(meta_value(meta, "cx"))
    radius = int(patchsize) // 2
    y = read_mha_2d(case_dir / fn_y)
    patch = y[cy - radius : cy + radius, cx - radius : cx + radius]
    if patch.shape != (patchsize, patchsize):
        raise ValueError(
            f"Reference patch shape {patch.shape}, expected {(patchsize, patchsize)} "
            f"for {case_dir}."
        )
    return patch


def robust_window(images: Iterable[np.ndarray], low: float, high: float) -> Tuple[float, float]:
    vals = np.concatenate([np.asarray(img, dtype=np.float32).ravel() for img in images])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1000.0, 2000.0
    vmin, vmax = np.percentile(vals, [float(low), float(high)])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return float(vals.min()), float(vals.max())
    return float(vmin), float(vmax)


def plot_grid(
    output: Path,
    columns: Sequence[Sequence[np.ndarray]],
    *,
    row_labels: Sequence[str],
    window: Tuple[float, float],
    dpi: int,
) -> None:
    n_rows = len(row_labels)
    n_cols = len(columns)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.05 * n_cols, 1.9 * n_rows),
        squeeze=False,
    )
    for col_idx, col in enumerate(columns):
        for row_idx, img in enumerate(col):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap="gray", vmin=window[0], vmax=window[1])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(f"Example {col_idx + 1}", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=9, rotation=90)
    fig.tight_layout(w_pad=0.12, h_pad=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    for subset in ("head", "body"):
        for method in METHODS:
            parser.add_argument(f"--{method}_{subset}_bank", required=True)
    parser.add_argument(
        "--example",
        action="append",
        required=True,
        help=(
            "Selected example as subset:index:sample_seed. Use multiple times. "
            "Example: --example body:717:1332"
        ),
    )
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fn_y", default="output_img.mha")
    parser.add_argument("--window_min_hu", type=float, default=-1000.0)
    parser.add_argument("--window_max_hu", type=float, default=2000.0)
    parser.add_argument("--window_low", type=float, default=1.0)
    parser.add_argument("--window_high", type=float, default=99.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    stats = read_stats(args.stats)
    banks: Dict[str, Dict[str, Dict[str, Any]]] = {"head": {}, "body": {}}
    for subset in ("head", "body"):
        for method in METHODS:
            banks[subset][method] = load_bank(getattr(args, f"{method}_{subset}_bank"))

    columns: List[List[np.ndarray]] = []
    all_images: List[np.ndarray] = []
    for spec in args.example:
        subset, patch_index, sample_seed = parse_example(spec)
        subset_banks = banks[subset]
        check_alignment(subset, patch_index, subset_banks)

        first_bank = subset_banks["pconv"]
        meta = first_bank["meta"][patch_index]
        fov = int(meta_value(meta, "fov", 0 if subset == "head" else 1))
        mean, std = mean_std(stats, fov)
        patchsize = int(first_bank["x"].shape[-1])

        input_patch = denorm(first_bank["x"][patch_index, 0], mean, std)
        ref_patch = reference_patch(meta, patchsize, args.fn_y)
        mask = np.squeeze(first_bank["m"][patch_index].detach().cpu().numpy()) > 0.5
        artifact_free_ref = np.where(mask, input_patch, ref_patch).astype(np.float32)

        samples = []
        for method_offset, method in enumerate(METHODS):
            bank = subset_banks[method]
            k = int(bank["samples"].shape[1])
            sidx = sample_index(k, sample_seed, patch_index, method_offset)
            samples.append(denorm(bank["samples"][patch_index, sidx, 0], mean, std))

        column = [input_patch, artifact_free_ref, *samples]
        columns.append(column)
        all_images.extend(column)

        case_id = Path(str(meta_value(meta, "case_dir"))).name
        print(
            f"{subset}:{patch_index}: sample_seed={sample_seed} case={case_id} "
            f"metal_pixels={int(mask.sum())}"
        )

    if args.window_min_hu is not None and args.window_max_hu is not None:
        window = (float(args.window_min_hu), float(args.window_max_hu))
    else:
        window = robust_window(all_images, args.window_low, args.window_high)

    plot_grid(
        Path(args.output),
        columns,
        row_labels=[
            "Input corrupted patch",
            "Artifact-free reference",
            "PConv-MAR sample",
            "CNN-MAR sample",
            "DuDoNet sample",
        ],
        window=window,
        dpi=args.dpi,
    )
    print(f"[written] {args.output}")


if __name__ == "__main__":
    main()
'''
python -u -m MAR_Inv.evaluation.plot_selected_mixed_invariance_examples \
  --pconv_body_bank results/sample_banks/test_banks/pconv_body_full_k1000_seed2026.pt \
  --cnnmar_body_bank results/sample_banks/test_banks/cnnmar_body_full_k1000_seed2026.pt \
  --dudonet_body_bank results/sample_banks/test_banks/dudonet_body_full_k1000_seed2026.pt \
  --pconv_head_bank results/sample_banks/test_banks/pconv_head_full_k1000_seed2026.pt \
  --cnnmar_head_bank results/sample_banks/test_banks/cnnmar_head_full_k1000_seed2026.pt \
  --dudonet_head_bank results/sample_banks/test_banks/dudonet_head_full_k1000_seed2026.pt \
  --example body:717:1332 \
  --example body:589:1332 \
  --example body:666:2026 \
  --example head:65:26 \
  --stats configs/head_body_stats_testset.json \
  --output results/invariance_samples_selected_examples_reproduced.png \
  --window_min_hu -1000 \
  --window_max_hu 2000 \
  --dpi 300
'''