""" Fan beam -> Parallel beam"""
import torch

def squeeze(x):
    if x.ndim == 4:
        if x.shape[1] != 1:
            raise ValueError("Expected a singleton channel dim.")
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(f"Expected [B,V,U], got shape={tuple(x.shape)}")
    return x


def pack_to_device(dc_pack, device):
    out = dict(dc_pack)
    for key, value in dc_pack.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
    return out


def rebin_proj(fan_beam_proj, pack):
    x = squeeze(fan_beam_proj)
    pack = pack_to_device(pack, x.device)
    view_count = pack["V"]
    det_cols = pack["U"]

    if x.shape[1] != view_count or x.shape[2] != det_cols:
        raise ValueError(f"Expected [B,{view_count},{det_cols}], got {tuple(x.shape)}")

    batch = x.shape[0]
    base = (torch.arange(batch, device=x.device) * (view_count * det_cols)).view(batch, 1, 1)
    zflat = x.reshape(-1)

    # the four fan-beam neighbourss of each parallel-beam bin
    g11 = zflat[base + pack["I11"]]
    g21 = zflat[base + pack["I21"]]
    g12 = zflat[base + pack["I12"]]
    g22 = zflat[base + pack["I22"]]

    return (
        pack["W11"] * g11
        + pack["W21"] * g21
        + pack["W12"] * g12
        + pack["W22"] * g22
    )
