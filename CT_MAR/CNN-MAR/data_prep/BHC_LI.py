"""BHC/LI input generation for CNN-MAR, plus the batch driver to run it.

CNN-MAR feeds the network three views of the same slice so it can learn to fuse
their complementary errors:
  * uncorrected (raw)  — has streaks but keeps true structure;
  * LI   — Linear-Interpolation MAR (removes streaks, may blur edges);
  * BHC  — a first-order Beam-Hardening Correction that models the metal's
           energy-dependent attenuation with a small polynomial and adds the
           correction back into the sinogram before reconstruction.

`marBHC_LI` computes the BHC and LI images from one raw sinogram + metal mask.
`write_bhc_li_img` (and the __main__ block) run it over a whole dataset split so
the 3-channel CNN-MAR inputs exist on disk before training. FOV (head vs body)
is chosen from the case id using the project-wide 1773 threshold.
"""
import os, numpy as np, SimpleITK as sitk
import argparse, yaml
from pathlib import Path
from types import SimpleNamespace
from src.mar_common.scanner_config import recon_config
from src.mar_common.io import read_mha, write_mha
from src.mar_common.cases import get_dir_names
from src.ct_ops.forward_proj import compute_sinogram
from src.ct_ops.fbp import reconstruct_fdk
from src.classical_mar.mar import mar_li

def marBHC_LI(proj, mask_dilated, fov):
    """
    Reduce metal artifacts using a first-order beam hardening correction (BHC).

    Parameters
    ----------
    proj: sinogram
    metal_proj : metal_trace

    Returns
    -------
    imBHC : (imPixNum, imPixNum) ndarray
        Beam-hardening corrected reconstruction (1/cm).
    imLi : (imPixNum, imPixNum) ndarray
        MAR-img.
    """
    views, bins = proj.shape
    N = views * bins
    cfg = recon_config(fov)
    metal_proj = compute_sinogram(mask_dilated, cfg)
    metal_trace = (metal_proj > 0).astype(proj.dtype)
    projLI = mar_li(proj, metal_trace)
    projDiff = proj - projLI
    projDiff1 = projDiff.reshape(N)
    metal_proj_real = metal_proj.reshape(N)
    metal_proj_bw1 = metal_trace.reshape(N).astype(float)
    A = np.zeros((N, 3), dtype=float)
    PM = metal_proj_bw1 * metal_proj_real
    A[:, 0] = PM
    A[:, 1] = PM * metal_proj_real
    A[:, 2] = PM * metal_proj_real ** 2
    y = metal_proj_bw1 * projDiff1
    X0, *_ = np.linalg.lstsq(A, y, rcond=None)
    projFit1 = A @ X0
    projDelta1 = metal_proj_bw1 * (X0[0] * metal_proj_real - projFit1)
    delta2d = projDelta1.reshape(views, bins)
    projBHC = proj + delta2d
    imBHC = reconstruct_fdk(cfg, projBHC)
    imLI = reconstruct_fdk(cfg, projLI)
    return (
     imBHC, imLI, projLI)

def _load_cfg(name):
    """Load configs/<name>.yaml from the repo root (or $MAR_CONFIG_DIR) as a namespace."""
    root = Path(os.environ.get(
        "MAR_CONFIG_DIR",
        Path(__file__).resolve().parents[2] / "configs",
    ))
    data = yaml.safe_load((root / f"{name}.yaml").read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{name}.yaml must contain a mapping")
    return SimpleNamespace(**data)

def write_bhc_li_img(main_dir):
    """For each case under main_dir, compute and save its BHC and LI input images."""
    folder_names = get_dir_names(main_dir)
    threshold = 1773  # case id <= 1773 => head FOV, else body FOV
    fov_head = 220.16
    fov_body = 400.0

    for i, foldername in enumerate(folder_names, start=0):
        proj = os.path.join(foldername, 'input_proj.mha')
        #metal_trace = os.path.join(foldername, 'input_metal_trace_proj.mha')
        metal_mask = os.path.join(foldername, 'input_mask_dilated.mha')

        proj = read_mha(proj)
        #metal_trace = read_mha(metal_trace)
        metal_mask = read_mha(metal_mask)

        folder_num = os.path.basename(foldername)
        try:
            case_id = int(folder_num)  # "01773" -> 1773
        except ValueError:
            print(f"  [skip] Non-numeric folder name: {foldername}")
            continue
        fov = fov_head if case_id <= threshold else fov_body

        print(f"foldername {i}: [{fov}]: {foldername}")

        img_bhc, img_li, _ = marBHC_LI(proj, metal_mask, fov)
        write_mha(img_bhc, os.path.join(foldername, 'img_bhc.mha'))
        write_mha(img_li, os.path.join(foldername, 'img_li.mha'))



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cnnmar",
                    help="Name of configs/<config>.yaml to load (or $MAR_CONFIG_DIR/<config>.yaml).")
    # Which dataset split to preprocess is set by `bh_li_dir` in the YAML.
    main_dir = _load_cfg(ap.parse_args().config).bh_li_dir
    write_bhc_li_img(main_dir)
'''
for val set, change bh_li_dir in the config.

python -u -m CNN-MAR.data_prep.BHC_LI --config cnnmar
'''