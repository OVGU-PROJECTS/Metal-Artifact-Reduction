"""Normalization stats for the 29 AAPM cases.
- mean over the head or body cases, std of each case, both on the rays outside the metal trace
- reads scoring29/, writes configs/aapm_norm_global_mean_case_std.json
"""
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk

root = Path(__file__).resolve().parents[3]
data, configs = root / "scoring29", root / "configs"
cases = ["%02d" % i for i in range(1, 30)]



def mha(path):
    return np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(path))))


# The reys the metal trace never touches
valid = {}
for c in cases:
    proj = mha(data / c / ("sino_%s.mha" % c)).astype(np.float64)
    valid[c] = proj[mha(data / c / ("metal_trace_%s.mha" % c)) == 0]

meta = json.loads((configs / "aapm_eval_cases.json").read_text())["cases"]
# head and body mean over all valid ray of their cases
group_mean = {a: float(np.concatenate([valid[c] for c in cases if meta[c]["norm"] == a]).mean())
              for a in ["head", "body"]}

# the pairing used for the submission folder
(configs / "aapm_norm_global_mean_case_std.json").write_text(json.dumps(
    {"mean_source": "aapm_29_head_body", "std_source": "case",
     "cases": {c: {"mean": group_mean[meta[c]["norm"]], "std": float(valid[c].std())}
               for c in cases}}, indent=2) + "\n")

print("done :)")

