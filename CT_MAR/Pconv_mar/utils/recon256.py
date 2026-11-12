"""Make "output_img_256.mha" in every case folder, the target of the PConv-MAR image losses
- FDK of "output_proj" with every 2nd view (1000 -> 500) on the 256 grid of config_downsampled

    python Pconv_mar/utils/recon256.py /mnt/public/bahaa/dataset_training
    python Pconv_mar/utils/recon256.py /mnt/public/bahaa/dataset_validation
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ct_ops.fbp import reconstruct_fdk
from geometry.fanbeam import RECO_FOV
from mar_common.cases import fov_from_folder, list_case_dirs
from mar_common.io import read_mha, write_mha
from mar_common.scanner_config import config_downsampled

# every 2nd view, like recon/cache.py ....
STRIDE = 2

configs = {name: config_downsampled(fov) for name, fov in RECO_FOV.items()}

root = Path(sys.argv[1])
cases = list_case_dirs(root)
print("found %d cases under %s" % (len(cases), root))

for case in tqdm(cases):
    out = case / "output_img_256.mha"
    if out.exists():
        continue
    proj = read_mha(str(case / "output_proj.mha")).astype(np.float32)
    cfg = configs["body" if fov_from_folder(case) == 1 else "head"]
    with contextlib.redirect_stdout(io.StringIO()): 
        img = reconstruct_fdk(cfg, proj[::STRIDE])
    write_mha(img.astype(np.float32), str(out))
