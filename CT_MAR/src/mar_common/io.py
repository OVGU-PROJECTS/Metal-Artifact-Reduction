import os
import numpy as np
import SimpleITK as sitk
import gecatsim as xc

import torch

def read_raw(raw_filename, num_slices, num_cols, num_rows):
    return xc.rawread(raw_filename, [num_slices, num_cols, num_rows], "float")


def read_proj_raw(raw_filename):
    proj = read_raw(raw_filename, 1000, 900, 1)
    return proj.squeeze()


def read_img_raw(raw_filename):
    img = read_raw(raw_filename, 1, 512, 512)
    return img.squeeze()


def write_mha(data, mha_filename):
    """Write a NumPy array to ``.mha`` (squeezed), creating the parent dir. """
    mha_filename = str(mha_filename)
    parent = os.path.dirname(mha_filename)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(np.asarray(data).squeeze()), mha_filename)


def read_mha(mha_filename):
    if not os.path.isfile(mha_filename):
        raise FileNotFoundError(mha_filename)
    data = sitk.GetArrayFromImage(sitk.ReadImage(mha_filename))
    return np.squeeze(data)


def write_raw(raw_filename, arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr.tofile(raw_filename)
  
    
def read_img(folder, fname):
    """Read "folder/fname" as a squeezed """
    return np.asarray(read_mha(os.path.join(folder, fname)), dtype=np.float32)


def to_4d_tensor(arr2d):
    """(H, W) NumPy array -> (1, 1, H, W) contiguous float32 torch tensor
    """
    t = torch.from_numpy(arr2d.astype(np.float32))
    return t.unsqueeze(0).unsqueeze(0).contiguous()
