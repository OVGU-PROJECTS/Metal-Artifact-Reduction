"""Shared by run_inference.py and predscoring.py.
- load the checkpoint
- run the network on one case: head or body
"""
import numpy as np
import torch

from mar_common.io import read_mha
from model.pconv_model import PConvUNet


def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint["model"]
    gm = state["fov_mean"].tolist()
    gs = state["fov_std"].tolist()
    model = PConvUNet(global_mean=gm, global_std=gs).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def predict_proj(model, proj_p, trace_p, mean, std, device):
    """network predcition for one case in mu units, and it valid-pixel mask"""
    proj = read_mha(str(proj_p)).astype(np.float32)
    trace = read_mha(str(trace_p)) > 0.5
    valid = 1.0 - trace.astype(np.float32)  # 1 = measured, 0 = metal trace

    proj = ((proj - mean) / (std + 1e-8)).astype(np.float32)
    proj[valid == 0] = 0.0

    x_proj = torch.from_numpy(proj[None, None, ...]).to(device)
    x_mask = torch.from_numpy(valid[None, None, ...]).to(device)
    with torch.inference_mode():
        y_pred = model(x_proj, x_mask)

    pred = y_pred[0, 0].detach().cpu().numpy().astype(np.float32)
    return pred * np.float32(std) + np.float32(mean), valid


