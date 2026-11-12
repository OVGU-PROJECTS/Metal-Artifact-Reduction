""" Analytical CNN-MAR post-processing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .prior import PriorInfo, build_prior_image
from .projection import interpolate_metal_trace


@dataclass(frozen=True)
class CNNMARPostprocessResult:
    """All intermediate stages."""

    image_hu: np.ndarray
    prior_hu: np.ndarray
    prior_projection: np.ndarray
    corrected_projection: np.ndarray
    metal_trace: np.ndarray
    prior_info: PriorInfo


def postprocess_cnnmar(
    measured_projection: np.ndarray,
    cnn_hu: np.ndarray,
    metal_mask: np.ndarray,
    reconstruction_config: Any,
    *,
    metal_trace: np.ndarray | None = None,
    forward_project: Callable[[np.ndarray, Any], np.ndarray] | None = None,
    reconstruct: Callable[[Any, np.ndarray], np.ndarray] | None = None,
) -> CNNMARPostprocessResult:
    """Create the CNN prior, correct the sinogram, and reconstruct the result.

    """
    if forward_project is None:
        from ct_ops.forward_proj import compute_sinogram

        forward_project = compute_sinogram
    if reconstruct is None:
        from ct_ops.fbp import reconstruct_fdk

        reconstruct = reconstruct_fdk

    measured = np.asarray(measured_projection)
    prior_hu, prior_info = build_prior_image(cnn_hu, metal_mask)
    mu_water = float(reconstruction_config.recon.mu)
    hu_offset = float(reconstruction_config.recon.huOffset)
    prior_mu = (prior_hu - hu_offset) * (mu_water / 1000.0)
    prior_projection = np.asarray(
        forward_project(prior_mu.astype(np.float32, copy=False),
                        reconstruction_config)
    )
    if prior_projection.shape != measured.shape:
        raise ValueError(
            "forward projection shape does not match measured projection: "
            f"{prior_projection.shape} != {measured.shape}"
        )

    if metal_trace is None:
        mask_projection = np.asarray(
            forward_project((np.asarray(metal_mask) > 0).astype(np.float32),
                            reconstruction_config)
        )
        trace = mask_projection > 0
    else:
        trace = np.asarray(metal_trace) > 0
    if trace.shape != measured.shape:
        raise ValueError(
            f"metal trace shape {trace.shape} does not match projection "
            f"shape {measured.shape}"
        )

    projection_difference = measured - prior_projection
    interpolated_difference = interpolate_metal_trace(
        projection_difference, trace, detector_axis=-1
    )
    corrected_projection = prior_projection + interpolated_difference
    corrected_projection = np.asarray(corrected_projection)
    corrected_projection[~trace] = measured[~trace]

    image_hu = np.asarray(
        reconstruct(reconstruction_config, corrected_projection),
        dtype=np.float32,
    )
    return CNNMARPostprocessResult(
        image_hu=image_hu,
        prior_hu=prior_hu,
        prior_projection=prior_projection,
        corrected_projection=corrected_projection,
        metal_trace=trace,
        prior_info=prior_info,
    )
