"""Projection-domain correction for CNN-MAR."""
from __future__ import annotations

import numpy as np


def interpolate_metal_trace(
    projection: np.ndarray,
    metal_trace: np.ndarray,
    *,
    detector_axis: int = -1,
) -> np.ndarray:
    """Linearly interpolate projection values inside the metal trace.

    Interpolation is performed independently for every view along detector
    bins.
    """
    values = np.asarray(projection)
    trace = np.asarray(metal_trace) > 0
    if values.ndim != 2 or trace.shape != values.shape:
        raise ValueError(
            f"projection and metal_trace must be equal-size 2-D arrays; "
            f"got {values.shape} and {trace.shape}"
        )

    moved = np.moveaxis(values, detector_axis, -1)
    moved_trace = np.moveaxis(trace, detector_axis, -1)
    result = moved.astype(np.result_type(values.dtype, np.float32), copy=True)
    bins = np.arange(result.shape[-1])

    for index in np.ndindex(result.shape[:-1]):
        invalid = moved_trace[index]
        if not np.any(invalid):
            continue
        valid = ~invalid
        if np.count_nonzero(valid) < 2:
            raise ValueError(
                "a projection view has fewer than two non-metal detector bins"
            )
        result[index][invalid] = np.interp(
            bins[invalid], bins[valid], result[index][valid]
        )

    return np.moveaxis(result, -1, detector_axis)
