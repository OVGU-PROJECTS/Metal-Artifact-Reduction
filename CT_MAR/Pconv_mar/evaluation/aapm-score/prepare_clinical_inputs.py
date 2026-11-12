"""Make the inputs of the 29 AAPM clinical cases.
- each case uses its own FOV and one fixed metal threshold

    python prepare_clinical_inputs.py --raw-dir ../phase3_final_test_data \
        --output-dir ../scoring29
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage


SCRIPT_DIR = Path(__file__).resolve().parent
CT_MAR_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(CT_MAR_ROOT / "src"))
sys.path.insert(0, str(CT_MAR_ROOT / "CNN-MAR"))

from data_prep.BHC_LI import marBHC_LI
from ct_ops.forward_proj import compute_sinogram
from mar_common.aapm_cases import load_eval_cases
from mar_common.io import write_mha
from mar_common.scanner_config import recon_config


RAW_DIR = CT_MAR_ROOT.parent / "phase3_final_test_data"
OUTPUT_DIR = CT_MAR_ROOT / "scoring29"
METAL_THRESHOLD_HU = 5000.0
TRACE_THRESHOLD = 0.001
NO_DILATION_CASES = {6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20}

def read_raw(path, shape):
    return np.fromfile(path, dtype="<f4").reshape(shape)

def prepare_case(raw_dir, output_dir, case, fov_mm, threshold_hu, dilate=True):
    image = read_raw(raw_dir / f"{case}.metalart.img_512x512x1.raw", (512, 512))
    sinogram = read_raw(raw_dir / f"{case}.metalart.sino_900x1000.raw", (1000, 900))

    mask = image > threshold_hu
    if dilate:
        mask_dilated = ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool))
    else:
        mask_dilated = mask
    metal_projection = compute_sinogram(mask_dilated.astype(np.float32), recon_config(fov_mm))
    metal_trace = metal_projection >= TRACE_THRESHOLD

    img_bhc, img_li, sino_li = marBHC_LI(sinogram, mask_dilated.astype(np.float32), fov_mm)

    case_dir = output_dir / case
    write_mha(image.astype(np.float32), case_dir / f"img_{case}.mha")
    write_mha(sinogram.astype(np.float32), case_dir / f"sino_{case}.mha")
    write_mha(mask.astype(np.uint8), case_dir / f"mask_{case}.mha")
    write_mha(mask_dilated.astype(np.uint8), case_dir / f"mask_dilated_{case}.mha")
    write_mha(metal_trace.astype(np.uint8), case_dir / f"metal_trace_{case}.mha")
    write_mha(np.asarray(sino_li, dtype=np.float32), case_dir / f"sino_li_{case}.mha")
    write_mha(np.asarray(img_li, dtype=np.float32), case_dir / f"img_li_{case}.mha")
    write_mha(np.asarray(img_bhc, dtype=np.float32), case_dir / f"img_bhc_{case}.mha")

    return int(mask.sum()), int(mask_dilated.sum()), int(metal_trace.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--threshold-hu", type=float, default=METAL_THRESHOLD_HU)
    args = parser.parse_args()

    cases = load_eval_cases()
    for case_id in range(1, 30):
        case = f"{case_id:02d}"
        fov_mm = float(cases[case]["fov_mm"])
        dilate = case_id not in NO_DILATION_CASES
        mask, dilated, trace = prepare_case(
            args.raw_dir, args.output_dir, case, fov_mm, args.threshold_hu, dilate
        )
        print(f"[{case}] FOV={fov_mm:.0f}mm mask={mask} "
              f"dilated={dilated}{'' if dilate else ' (no dilation)'} trace={trace}", flush=True)

    print(f"[done] wrote {args.output_dir}")



if __name__ == "__main__":
    main()


