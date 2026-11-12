"""
Code in this file is partially or fully based on, and in some cases adapted from, the implementation provided in the following repository:https://github.com/eeulig/ldct-invariances
"""


# helper utilities used across the training code.
# Includes YAML config loading, pickle serialization, and simple image windowing.
import os
import pickle
from typing import Tuple, Union

import numpy as np
import yaml

# Helper utilities for config, serialization, and simple image windowing.

def load_yaml(path):
    """Load YAML content from a file"""
    with open(path) as file:
        content = yaml.load(file, Loader=yaml.FullLoader)
    return content


def dump_config(args, path):
    """write args to args.yaml for reproducible experiment logging."""
    with open(os.path.join(path, "args.yaml"), "w") as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)


def save_obj(obj, path):
    """Save Python object to disk with pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    """Load a Python object previously saved with pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_windowing(
    x: np.ndarray,
    center: Union[int, float, None] = None,
    width: Union[int, float, None] = None,
    out_range: Tuple = (0, 1),
) -> np.ndarray:
    """Apply intensity windowing to image and scale into out_range."""
    # Both center and width must be provided to define a valid window.
    if center is None and width is None:
        raise ValueError("Center and width must be provided.")

    center = float(center)
    width = float(width)

    lower = center - 0.5 - (width - 1) / 2.0
    upper = center - 0.5 + (width - 1) / 2.0
    # Allocate output array with the same shape and dtype as the input.
    res = np.empty(x.shape, dtype=x.dtype)
    res[x <= lower] = out_range[0]
    res[(x > lower) & (x <= upper)] = ((x[(x > lower) & (x <= upper)] - (center - 0.5)) / (width - 1.0) + 0.5) * (
        out_range[1] - out_range[0]
    ) + out_range[0]
    res[x > upper] = out_range[1]
    return res
