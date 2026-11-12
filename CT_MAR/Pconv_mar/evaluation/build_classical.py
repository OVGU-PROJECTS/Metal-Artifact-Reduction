"""LI-MAR and NMAR on the test set, in the layout score_images.py reads.

    python -m evaluation.build_classical --fov head
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL.parents[1] / "src"))

from classical_mar.nmar import PRIOR_EPS, mar_li, nmar_prior, nmar_thresholds, proj_interp
from ct_ops.fbp import reconstruct_fdk
from ct_ops.forward_proj import compute_sinogram
from geometry.fanbeam import RECO_FOV
from mar_common.cases import list_case_dirs
from mar_common.io import read_mha
from mar_common.scanner_config import recon_config
from mar_common.units import hu2mu

TEST_DIR = EVAL.parents[2] / "test_images_1000"
REF_DIR = EVAL / "predictions" / "pconv_fullstack_bestckpt"  # target/ and mask/ come from here
OUT_DIR = EVAL / "classical_baselines_predictions"


def nmar(proj, trace, metal, config):
    """NMAR; views whose edge rays miss the patient fall back to LI."""
    mu_water = config.recon.mu
    proj_li, img_li = mar_li(proj, trace, config)

    img_mu = hu2mu(np.asarray(img_li, np.float64), mu_water)
    img_mu[metal] = mu_water
    prior_mu = nmar_prior(img_mu, *nmar_thresholds(img_mu, mu_water), mu_water)
    prior = np.maximum(np.asarray(compute_sinogram(prior_mu, config), np.float64), 0) + PRIOR_EPS

    out = proj_interp(proj / prior, trace) * prior
    edge = ndimage.binary_dilation(trace, np.ones((1, 3))) & ~trace
    bad = (edge & (prior < 0.1)).any(axis=1)
    out[bad] = proj_li[bad]
    out[~trace] = proj[~trace]
    return reconstruct_fdk(config, out)


def run_case(job):
    case, fov, name = job
    proj = read_mha(str(case / "input_proj.mha")).astype(np.float64)
    metal = read_mha(str(case / "input_mask.mha")) > 0.5
    trace = read_mha(str(case / "input_metal_trace_proj.mha")) > 0
    config = recon_config(RECO_FOV[fov])

    img_li = mar_li(proj, trace, config)[1]
    img_nmar = nmar(proj, compute_sinogram(metal, config) > 1e-6, metal, config)  # metal-only trace

    np.save(OUT_DIR / "LI-MAR" / fov / "pred" / name, np.asarray(img_li, np.float32))
    np.save(OUT_DIR / "nmar" / fov / "pred" / name, np.asarray(img_nmar, np.float32))
    print(fov, name, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fov", required=True, choices=["head", "body"])
    fov = ap.parse_args().fov

    for method in ("LI-MAR", "nmar"):
        root = OUT_DIR / method / fov
        (root / "pred").mkdir(parents=True, exist_ok=True)
        for sub in ("target", "mask"):
            if not (root / sub).is_symlink():
                (root / sub).symlink_to(REF_DIR / fov / sub)

    cases = list_case_dirs(TEST_DIR / fov)
    jobs = [(case, fov, f"nometal_{i:04d}_512x512.npy") for i, case in enumerate(cases, 1)]
    with Pool(8) as pool:
        pool.map(run_case, jobs)


if __name__ == "__main__":
    main()
