"""MATLAB-compatible patch generation for CNN-MAR training.

This module ports ``cnnmar/dataGen.m`` from the original Zhang/Yu release.
For every case it evaluates all patches on a regular grid, takes
disjoint maximum-error patches for each of the three input channels, and fills
the remaining quota by sampling uniformly from the unselected grid positions.
"""
from __future__ import annotations

import math
from typing import Iterator, Optional, Sequence

import numpy as np

from mar_common.cases import get_dir_names
from mar_common.io import read_img
from data_pipline.preprocessing import prepare_training_images


_FILENAMES = {
    "raw": "input_img.mha",
    "bhc": "img_bhc.mha",
    "li": "img_li.mha",
    "ref": "output_img.mha",
    "mask": "input_mask.mha",
}


def _matlab_round_positive(value: float) -> int:
    """Match MATLAB's round for the non-negative quotas used by dataGen.m."""
    return int(math.floor(value + 0.5))


def _grid_patch_sse(diff: np.ndarray, patch: int, stride: int):
    """Return grid starts and per-channel patch sums of squared differences."""
    h, w, channels = diff.shape
    ys = np.arange(0, h - patch, stride, dtype=np.int64)
    xs = np.arange(0, w - patch, stride, dtype=np.int64)
    if ys.size == 0 or xs.size == 0:
        raise ValueError(
            f"Image {diff.shape[:2]} is too small for patch={patch}, stride={stride}"
        )

    squared = np.square(diff, dtype=np.float64)
    integral = np.pad(
        np.cumsum(np.cumsum(squared, axis=0), axis=1),
        ((1, 0), (1, 0), (0, 0)),
    )
    y0 = ys[:, None]
    x0 = xs[None, :]
    y1 = y0 + patch
    x1 = x0 + patch
    scores = (
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    )

    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    coords = np.column_stack((grid_y.ravel(), grid_x.ravel()))
    return coords, scores.reshape(-1, channels)


def select_original_patch_coords(
    raw: np.ndarray,
    bhc: np.ndarray,
    li: np.ndarray,
    ref: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
    patch: int = 64,
    stride: int = 8,
    max_diff_ratio: float = 0.6,
) -> np.ndarray:
    """Select patch follows the methodology used in the original implementation in ``dataGen.m``.

    The maximum-error selections are made sequentially for raw, BHC, and LI.
    A location selected for one channel is removed before the next channel is
    ranked, so every returned coordinate is unique.
    """
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if patch < 1 or stride < 1:
        raise ValueError(f"patch and stride must be positive, got {patch}, {stride}")
    if not 0.0 <= max_diff_ratio <= 1.0:
        raise ValueError(f"max_diff_ratio must be in [0, 1], got {max_diff_ratio}")

    inputs = np.stack((raw, bhc, li), axis=-1)
    if inputs.shape[:2] != ref.shape[:2]:
        raise ValueError(f"Input/reference shape mismatch: {inputs.shape[:2]} vs {ref.shape[:2]}")

    coords, scores = _grid_patch_sse(inputs - ref[..., None], patch, stride)
    if count > len(coords):
        raise ValueError(
            f"Requested {count} unique patches, but only {len(coords)} grid positions "
            f"exist for image={ref.shape}, patch={patch}, stride={stride}"
        )

    max_diff_count = _matlab_round_positive(count * max_diff_ratio)
    per_channel = _matlab_round_positive(max_diff_count / inputs.shape[-1])
    strongest_total = per_channel * inputs.shape[-1]
    if strongest_total > count:
        raise ValueError(
            "MATLAB quota rounding selected more strong patches than the total: "
            f"count={count}, max_diff_ratio={max_diff_ratio}"
        )

    remaining = np.arange(len(coords), dtype=np.int64)
    selected: list[np.ndarray] = []
    for channel in range(inputs.shape[-1]):
        if per_channel == 0:
            break
        # Stable descending order gives deterministic tie handling.
        order = np.argsort(-scores[remaining, channel], kind="stable")
        chosen_positions = order[:per_channel]
        selected.append(remaining[chosen_positions])
        keep = np.ones(remaining.size, dtype=bool)
        keep[chosen_positions] = False
        remaining = remaining[keep]

    random_count = count - strongest_total
    if random_count:
        random_indices = rng.permutation(remaining)[:random_count]
        selected.append(random_indices)

    return coords[np.concatenate(selected)]


def _load_prepared_case(folder: str):
    raw = read_img(folder, _FILENAMES["raw"])
    bhc = read_img(folder, _FILENAMES["bhc"])
    li = read_img(folder, _FILENAMES["li"])
    ref = read_img(folder, _FILENAMES["ref"])
    metal = read_img(folder, _FILENAMES["mask"])
    return prepare_training_images(raw, bhc, li, ref, metal)


def _iter_original_cycle(
    folders: Sequence[str],
    total_patches: int,
    rng: np.random.Generator,
    *,
    patch: int,
    stride: int,
    max_diff_ratio: float,
    shuffle: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    case_order = list(folders)
    if shuffle:
        rng.shuffle(case_order)
    patches_per_case = math.ceil(total_patches / len(case_order))
    produced = 0

    for folder in case_order:
        raw, bhc, li, ref = _load_prepared_case(folder)
        coords = select_original_patch_coords(
            raw,
            bhc,
            li,
            ref,
            count=patches_per_case,
            rng=rng,
            patch=patch,
            stride=stride,
            max_diff_ratio=max_diff_ratio,
        )
        if shuffle:
            rng.shuffle(coords)

        for y0, x0 in coords:
            x = np.stack(
                (
                    raw[y0:y0 + patch, x0:x0 + patch],
                    bhc[y0:y0 + patch, x0:x0 + patch],
                    li[y0:y0 + patch, x0:x0 + patch],
                ),
                axis=-1,
            ).astype(np.float32, copy=False)
            y = ref[y0:y0 + patch, x0:x0 + patch, None].astype(np.float32, copy=False)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"Non-finite value in case folder: {folder}")
            yield x, y
            produced += 1
            if produced == total_patches:
                return


def gen_cnnmar(
    max_examples: int = -1,
    batch_size: int = 16,
    data_t: str = "train",
    seed: Optional[int] = None,
    shuffle: bool = True,
    reshuffle_each_epoch: bool = True,
    *,
    train_root: str,
    val_root: str,
    test_root: str,
    patches_per_cycle: Optional[int] = None,
    patch: int = 64,
    strong_ratio: float = 0.6,
    stride: int = 8,
):
    """Yield batches made with the original CNN-MAR patch sampler.

    Following the original implementaion, the selected patches are materialized once. Training can then globally reshuffle those same patches at
    every epoch without changing patch membership.
    """
    split = data_t if isinstance(data_t, str) else data_t.decode()
    try:
        base_dir = {"train": train_root, "val": val_root, "test": test_root}[split]
    except KeyError as exc:
        raise ValueError(f"Unknown split {split!r}; expected train, val, or test") from exc

    folders = sorted(get_dir_names(base_dir))
    if not folders:
        raise ValueError(f"No case folders found in {base_dir}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    finite = max_examples is not None and max_examples > 0
    if finite:
        total_patches = int(max_examples)
    elif patches_per_cycle is not None and patches_per_cycle > 0:
        total_patches = int(patches_per_cycle)
    else:

        total_patches = len(folders)

    fixed_seed = seed if seed is not None else int(np.random.SeedSequence().entropy)

    x_all = np.empty((total_patches, patch, patch, 3), dtype=np.float32)
    y_all = np.empty((total_patches, patch, patch, 1), dtype=np.float32)
    selection_rng = np.random.default_rng(fixed_seed)
    for index, (x, y) in enumerate(_iter_original_cycle(
        folders,
        total_patches,
        selection_rng,
        patch=patch,
        stride=stride,
        max_diff_ratio=strong_ratio,
        shuffle=False,
    )):
        x_all[index] = x
        y_all[index] = y

    order_rng = np.random.default_rng(fixed_seed)
    fixed_order = order_rng.permutation(total_patches) if shuffle else np.arange(total_patches)
    while True:
        if shuffle and reshuffle_each_epoch:
            order = order_rng.permutation(total_patches)
        else:
            order = fixed_order
        for start in range(0, total_patches, batch_size):
            batch_indices = order[start:start + batch_size]
            yield x_all[batch_indices], y_all[batch_indices]
        if finite:
            return
