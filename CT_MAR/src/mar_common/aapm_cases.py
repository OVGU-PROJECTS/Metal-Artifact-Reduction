"""The 29 clinical cases of the AAPM CT-MAR challenge.
- reads configs/aapm_eval_cases.json: FOV, metal HU and head/body stats of each case
- used by the AAPM submission scirpts of PConv-MAR, DuDoNet and the CNN
"""
import json
import os
import re
from pathlib import Path


def default_cases_meta():
    """configs/aapm_eval_cases.json, or the one in MAR_CONFIG_DIR if it is set"""
    configs = Path(os.environ.get("MAR_CONFIG_DIR", Path(__file__).resolve().parents[2] / "configs"))
    return configs / "aapm_eval_cases.json"


def load_eval_cases(path=None):
    """case id ("01" to "29") -> {"fov_mm", "metal_hu", "norm"}"""
    if path is None:
        path = default_cases_meta()
    return json.loads(Path(path).read_text())["cases"]



def case_id(case_dir):
    """last number in the folder name as two digits, e.g. "case_7" -> "07"."""
    nums = re.findall(r"\d+", Path(case_dir).name)
    if not nums:
        raise ValueError(f"No case id in folder name: {case_dir}")
    return f"{int(nums[-1]):02d}"


















