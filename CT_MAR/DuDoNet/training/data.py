"""Batch loading for DuDoNet training.  """
import torch

from mar_common.io import read_mha, to_4d_tensor
from mar_common.units import hu2mu
from mar_common.cases import fov_from_folder


def load_batch(case_batch, device="cuda"):
    """Load case directories into model tensors.
    Returns:
      y_li:  (B,1,1000,900)
      mt:    (B,1,1000,900)
      y_gt:  (B,1,1000,900)
      x_gt:  (B,1,512,512)
      fov:   (B,) int64 0=head,1=body
    """

    y_li_list, mt_list, ygt_list, xgt_list, fov_list = [], [], [], [], []

    for case_dir in case_batch:
        fov = fov_from_folder(str(case_dir))

        # Dataset files.
        y_li = read_mha(str(case_dir / "input_proj_li.mha"))          # (1000,900)
        mt   = read_mha(str(case_dir / "input_metal_trace_proj.mha")) # (1000,900)
        y_gt = read_mha(str(case_dir / "output_proj.mha"))            # (1000,900)
        x_gt = read_mha(str(case_dir / "output_img.mha"))             # (512,512)
        x_gt = hu2mu(x_gt)

        y_li_list.append(to_4d_tensor(y_li))
        mt_list.append(to_4d_tensor(mt))
        ygt_list.append(to_4d_tensor(y_gt))
        xgt_list.append(to_4d_tensor(x_gt))
        fov_list.append(fov)

    y_li = torch.cat(y_li_list, dim=0).to(device, non_blocking=True)
    mt   = torch.cat(mt_list,   dim=0).to(device, non_blocking=True)
    y_gt = torch.cat(ygt_list,  dim=0).to(device, non_blocking=True)
    x_gt = torch.cat(xgt_list,  dim=0).to(device, non_blocking=True)
    fov  = torch.tensor(fov_list, dtype=torch.long, device=device)

    return y_li, mt, y_gt, x_gt, fov
