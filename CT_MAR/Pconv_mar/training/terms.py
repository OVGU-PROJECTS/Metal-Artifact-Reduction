"""The six loss terms and the config keys that carry them.
those two tables stay explicit rather than being retyped in
every script that reads or writes a config.

Imports nothing, so config-only tools can use it without pulling in torch.
"""

# report order for the calibration and gradient tools
TERMS = ("rmse", "hole", "dc", "img", "edge", "perc")

CALIB_KEY = {term: f"{term}_scale" for term in TERMS}

WEIGHT_KEY = {
    "rmse": "w_rmse",
    "hole": "w_hole",
    "dc": "w_dc",
    "img": "w_image",
    "edge": "w_edge",
    "perc": "w_perc",
}
# rmse carries no start epoch, it is active from the first step
START_KEY = {
    "hole": "start_ep_hole",
    "dc": "start_ep_dc",
    "img": "start_ep_image",
    "edge": "start_ep_edge",
    "perc": "start_ep_perc",
}

