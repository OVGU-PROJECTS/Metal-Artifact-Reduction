"""Dataset for the partial-conv sinogram model.
Each item gives the network:
1- a metal-masked sinogram (holes zeroed) its metal trace mask
2- The training target packs two things: the clean sinogram, plus an "img_pack"
that carries the 256x256 ground-truth image (for image loss term) with
the head/body FOV code tucked into its bottom-right pixel.
- Sinograms are z-scored per FOV using precomputed global mean/std.
"""
import os
import numpy as np
import torch      
from torch.utils.data import DataLoader, Dataset 
from torch.utils.data.distributed import DistributedSampler

from mar_common.io import read_mha
from mar_common.cases import get_dir_names, fov_from_folder
from mar_common.units import hu2mu 

ROWS = 1000
COLS = 900

class ProjDataset(Dataset):
    """yields ((masked_proj, metal trace), packed_target) per case
    Shapes:
    masked_proj:   (1, 1000, 900)
    mask:          (1, 1000, 900)
    packed_target: (2, 1000, 900)
        channel 0: ground truth sinogram
        channel 1: 256x256 gound truth image in top-left, FOV code at [-1, -1]
    """
    def __init__(
        self,
        global_mean,
        global_std,
        base_dir,
        fixed_case_shuffle= False,
        fixed_case_seed= 0,
    ):
        self.global_mean = np.asarray(global_mean, dtype=np.float32)
        self.global_std = np.asarray(global_std, dtype=np.float32)

        self.case_paths = get_dir_names(base_dir)
        if fixed_case_shuffle:
            rng = np.random.default_rng(int(fixed_case_seed))
            rng.shuffle(self.case_paths)

    def __len__(self):
        return len(self.case_paths)

    def __getitem__(self, index):
        folder = self.case_paths[index]

        proj = read_mha(os.path.join(folder, "input_proj.mha")).astype(np.float32)
        label = read_mha(os.path.join(folder, "output_proj.mha")).astype(np.float32)

        # z-score normalization
        fov_code = fov_from_folder(folder)
        m = float(self.global_mean[fov_code])
        s = float(self.global_std[fov_code])
        proj = (proj - m) / (s + 1e-8)
        label = (label - m) / (s + 1e-8)

        # mask = 1 where the ray is trustworthy, 0 inside the metal trace (the "hole").
        trace = read_mha(os.path.join(folder, "input_metal_trace_proj.mha"))
        mask = (trace < 0.5).astype(np.float32)

        proj_masked = proj * mask  # zero out the metal trace region

        # a passthrough ture image used in image loss
        img_true256 = read_mha(os.path.join(folder, "output_img_256.mha"))
        img_true256 = hu2mu(img_true256).astype(np.float32)

        # add the fov codes to this image
        img_pack = np.zeros((ROWS, COLS), dtype=np.float32)
        img_pack[:256, :256] = img_true256
        img_pack[-1, -1] = fov_code

        x_proj = torch.from_numpy(proj_masked[None, ...])
        x_mask = torch.from_numpy(mask[None, ...])

        y_true = np.stack([label, img_pack], axis=0).astype(np.float32)
        y_true = torch.from_numpy(y_true)
        return (x_proj, x_mask), y_true


def build_loader(
    batch_size,
    global_mean,
    global_std,
    shuffle: bool,
    num_workers,
    base_dir,
    fixed_case_shuffle= False,
    fixed_case_seed = 0,
    distributed= False,
    rank= 0,
    world_size = 1,
    seed =0,
):
    ds = ProjDataset(
        global_mean=global_mean,
        global_std=global_std,
        fixed_case_shuffle=fixed_case_shuffle,
        fixed_case_seed=fixed_case_seed,
        base_dir=base_dir,
    )
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    ), sampler
