"""FDK setups for the metrics stage: SSIM/MAE scoring, not the training loss.
- used in training/evaluation + evaluation/*
- pconv_model.py runs at 500 views on a downsampled config.
- The evaluation metrics are defined on the full 512x512 / 1000-view
reconstruction, so they need their own setup, built here.
"""
from functools import lru_cache

from mar_common.scanner_config import recon_config

from recon.cache import RECO_FOV, createFdkCache
from recon.filtering import load_createhsp


@lru_cache(maxsize=None)
def metric_fdk_setups():
    """{fov_name: (config, ramp filter)} for head and body, built once per process"""
    context = {}
    for fov_name in ("head", "body"):
        cfg = recon_config(RECO_FOV[fov_name])
        cfg._fdk_cache = createFdkCache(cfg, target_views=1000, view_block=50, raw_view_count=1000)
        context[fov_name] = (cfg, load_createhsp(fov_name))
    return context
