import os
import json
import re
from argparse import Namespace
from multiprocessing import Value
from typing import Any, Dict, Optional, Tuple

from MAR_Inv.utils.file_info import get_dir_names
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset for VAE and cINN inputs where each case contributes a fixed number of patches per epoch.
# Supports unconditional loading of input patches and optional conditional loading

class PatchesGenerator(Dataset):
    """
    Each case contributes `patches_per_image` patches per epoch.

    Always returns:
      - x: (1,P,P) input patch (z-score normalized with head/body stats)

    If conditional=True, also returns:
      - m: (1,P,P) metal mask patch (0/1)
      - p: (1,P,P) pred patch (z-score normalized with head/body stats)

    If conditional=False:
      - does not require mask/pred files
      - by default does not return 'm'/'p'

    Patch positions change each epoch (even with num_workers>0) using shared epoch value.
    """

    def __init__(
        self,
        args: Namespace,
        *,
        fn_x: str = "input_img.mha",
        fn_y: Optional[str] = "output_img.mha",   # used only for strength sampling if available
        fn_mask: str = "input_mask.mha",
        fn_pred: Optional[str] = None,            # default built from args.greybox (conditional only)
        patches_per_image: Optional[int] = None,
        strong_ratio: float = 0.7,
        topk: int = 5000,
        use_strength: bool = True,
        return_meta: bool = True,
        stats_path: str = "/mnt/public/bahaa/our_model/head_body_stats.json",
        conditional: Optional[bool] = None,       # if None -> reads args.conditional
        require_cond_files: bool = True,          # if conditional=True and files missing -> error
    ):
        super().__init__()

        self.seed = int(getattr(args, "seed", 0))
        self.patchsize = int(getattr(args, "patchsize", 128))
        self.data_subset = float(getattr(args, "data_subset", 1.0))
        self.data_norm = str(getattr(args, "data_norm", "meanstd"))
        if self.data_norm != "meanstd":
            raise ValueError("This dataset expects data_norm='meanstd'.")

        self.datafolder = str(getattr(args, "datafolder", ""))
        if not self.datafolder:
            raise ValueError("args.datafolder must be set (no train/val resolving here).")

        if conditional is None:
            conditional = bool(getattr(args, "conditional", True))
        self.conditional = bool(conditional)
        self.require_cond_files = bool(require_cond_files)

        self.fn_x = str(fn_x)
        self.fn_y = str(fn_y) if fn_y is not None else None

        # conditional filenames (only needed if conditional=True)
        self.fn_mask = str(fn_mask) if self.conditional else None


        if self.conditional:
            # pred file name: pred_<name>_img.mha
            if fn_pred is None:
                pred_name = getattr(args, "pred_name", None)
                if not pred_name:
                    pred_name = getattr(args, "greybox", None)
                if not pred_name:
                    raise ValueError("Set args.pred_name (e.g. 'cnnmar') to load pred_<name>_img.mha, or pass fn_pred.")
                fn_pred = f"pred_{pred_name}_img.mha"
            self.fn_pred = str(fn_pred)
        else:
            self.fn_pred = None

        if patches_per_image is None:
            patches_per_image = int(getattr(args, "patches_per_image", 1))
        self.patches_per_image = int(patches_per_image)
        if self.patches_per_image <= 0:
            raise ValueError("patches_per_image must be >= 1")

        self.strong_ratio = float(strong_ratio)
        self.topk = int(topk)
        self.use_strength = bool(use_strength)
        self.return_meta = bool(return_meta)

        # ---- load stats JSON ----
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"Stats JSON not found: {stats_path}")
        with open(stats_path, "r", encoding="utf-8") as f:
            st = json.load(f)

        self.head_last_id = int(st["head_last_id"])
        self.mask_threshold = float(st.get("mask_threshold", 0.5))
        self.mean_head = float(st["head"]["mean"])
        self.std_head = float(st["head"]["std"])
        self.mean_body = float(st["body"]["mean"])
        self.std_body = float(st["body"]["std"])

        # shared epoch for num_workers>0
        # Shared epoch value lets multiple workers sample patches consistently across epochs.
        self._epoch = Value("i", 0)

        # case folders
        self.case_dirs = sorted(get_dir_names(self.datafolder))
        if not self.case_dirs:
            raise ValueError(f"No case folders found in: {self.datafolder}")

        if self.data_subset < 1.0:
            n = max(1, int(len(self.case_dirs) * self.data_subset))
            self.case_dirs = self.case_dirs[:n]

    # -------------------------
    # epoch control
    # -------------------------
    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def get_epoch(self) -> int:
        with self._epoch.get_lock():
            return int(self._epoch.value)

    # -------------------------
    # basic utils
    # -------------------------
    def __len__(self) -> int:
        return len(self.case_dirs) * self.patches_per_image

    @staticmethod
    def _to_torch_1ch(x: np.ndarray) -> torch.Tensor:
        # x: (H,W) float32 -> (1,H,W)
        return torch.from_numpy(x.astype(np.float32, copy=False)).unsqueeze(0)

    @staticmethod
    def read_mha(mha_filename: str) -> np.ndarray:
        """Read .mha file -> (H,W) numpy array (squeezed)."""
        if not os.path.isfile(mha_filename):
            raise FileNotFoundError(f"Missing file: {mha_filename}")
        data = sitk.GetArrayFromImage(sitk.ReadImage(mha_filename))
        return np.squeeze(data)

    @staticmethod
    def _zscore(x: np.ndarray, mean: float, std: float, eps: float = 1e-6) -> np.ndarray:
        return (x - mean) / (std + eps)

    @staticmethod
    def _case_id_from_folder(case_dir: str) -> Optional[int]:
        base = os.path.basename(os.path.normpath(case_dir))
        m = re.search(r"(\d+)", base)
        return int(m.group(1)) if m else None

    def _stats_for_case(self, case_dir: str) -> Tuple[float, float, int]:
        """
        Returns (mean, std, fov) where fov: 0=head, 1=body
        Rule: if folder number > head_last_id => body
        """
        cid = self._case_id_from_folder(case_dir)
        if cid is None:
            return self.mean_body, self.std_body, 1
        if cid <= self.head_last_id:
            return self.mean_head, self.std_head, 0
        return self.mean_body, self.std_body, 1

    def _sample_center_from_strength(
        self, strength: np.ndarray, rng: np.random.Generator, P: int, strong: bool
    ) -> Tuple[int, int]:
        """Choose a patch center based on a strength map.

        If `strong` is True, select from the top-k strongest pixels inside the valid region.
        Otherwise, select a random valid center.
        """
        h, w = strength.shape
        r = P // 2
        if h < P or w < P:
            return h // 2, w // 2

        sv = strength[r : h - r, r : w - r]  # valid centers
        flat = sv.reshape(-1)
        n = flat.size
        if n == 0:
            return h // 2, w // 2

        if strong:
            k = max(1, min(self.topk, n))
            idxs = np.argpartition(flat, -k)[-k:]
            pick = int(rng.choice(idxs))
        else:
            pick = int(rng.integers(0, n))

        yy, xx = divmod(pick, sv.shape[1])
        return int(yy + r), int(xx + r)

    @staticmethod
    def _fallback_strength_from_x(x: np.ndarray) -> np.ndarray:
        """
        Strength map using gradient magnitude of x (works without GT y).
        """
        x_f = x.astype(np.float32, copy=False)
        gy, gx = np.gradient(x_f)
        return (np.abs(gx) + np.abs(gy)).astype(np.float32, copy=False)

    # -------------------------
    # main sample
    # -------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        n_cases = len(self.case_dirs)
        case_idx = int(idx // self.patches_per_image) % n_cases
        patch_j  = int(idx % self.patches_per_image)
        case_dir = self.case_dirs[case_idx]

        epoch = self.get_epoch()

        # epoch in seed => different patches per epoch, stable across workers
        seed = (self.seed * 1_000_003) + (epoch * 1_000_000_007) + (case_idx * 10_000) + patch_j
        rng = np.random.default_rng(seed)

        case_path = case_dir
        # if get_dir_names() returns folder names (not absolute paths), resolve them
        if not os.path.isabs(case_path):
            case_path = os.path.join(self.datafolder, case_path)

        # The input image
        x = self.read_mha(os.path.join(case_path, self.fn_x))

        # optional conditioning loads
        m = None
        p = None

        if self.conditional:
            # Load mask and prediction files only when conditional mode is enabled.
            mask_path = os.path.join(case_path, self.fn_mask)
            pred_path = os.path.join(case_path, self.fn_pred)

            if not os.path.isfile(mask_path) or not os.path.isfile(pred_path):
                if self.require_cond_files:
                    raise FileNotFoundError(
                        f"Conditional=True but missing mask or pred in {case_dir}"
                    )

            m = self.read_mha(mask_path)
            p = self.read_mha(pred_path)

            if m.shape != x.shape or p.shape != x.shape:
                raise ValueError(f"Shape mismatch in {case_dir}")

            m = (m > self.mask_threshold).astype(np.float32)

            # restore metal values from corrupted input x into prediction p
            # metal region comes from x, non-metal region stays from p
            p = np.where(m > 0, x, p).astype(np.float32, copy=False)  


        # --- choose patch center ---
        P = self.patchsize
        r = P // 2
        strong = bool(rng.random() < self.strong_ratio)

        if self.use_strength:
            # Preferred: |x - y| if y exists; otherwise fallback to gradients of x
            strength = None
            if self.fn_y is not None:
                y_path = os.path.join(case_path, self.fn_y)
                if os.path.isfile(y_path):
                    y = self.read_mha(y_path)
                    if y.shape != x.shape:
                        raise ValueError(f"Shape mismatch in {case_dir}: x{tuple(x.shape)} y{tuple(y.shape)}")
                    strength = np.abs(x - y).astype(np.float32, copy=False)

            if strength is None:
                strength = self._fallback_strength_from_x(x)

            cy, cx = self._sample_center_from_strength(strength, rng, P, strong=strong)
        else:
            h, w = x.shape
            if h < P or w < P:
                cy, cx = h // 2, w // 2
            else:
                cy = int(rng.integers(r, h - r))
                cx = int(rng.integers(r, w - r))

        # --- extract patches ---
        # The patch is centered at the chosen coordinates and has size P x P.
        x_p = x[cy - r : cy + r, cx - r : cx + r]

        if self.conditional:
            assert m is not None and p is not None
            m_p = m[cy - r : cy + r, cx - r : cx + r]
            p_p = p[cy - r : cy + r, cx - r : cx + r]
        else:
            m_p = None
            p_p = None

        # final guard if image smaller than patch
        if x_p.shape != (P, P):
            x_p = x_p[:P, :P]
            if m_p is not None:
                m_p = m_p[:P, :P]
            if p_p is not None:
                p_p = p_p[:P, :P]

        # Clip extreme values and convert to float32 before normalization.
        x_p = np.clip(x_p, x_p.min(), 3000).astype(np.float32, copy=False)
        if p_p is not None:
            p_p = np.clip(p_p, p_p.min(), 3000).astype(np.float32, copy=False)

        # Choose head/body normalization stats based on case folder.
        mean, std, fov = self._stats_for_case(case_dir)

        # z-score normalize x (and pred if present)
        x_pn = self._zscore(x_p, mean, std).astype(np.float32, copy=False)

        out: Dict[str, Any] = {"x": self._to_torch_1ch(x_pn)}

        if self.conditional:
            assert m_p is not None and p_p is not None
            p_pn = self._zscore(p_p, mean, std).astype(np.float32, copy=False)
            out.update(
                {
                    "m": self._to_torch_1ch(m_p.astype(np.float32, copy=False)),
                    "p": self._to_torch_1ch(p_pn),
                }
            )

        # Metadata is optional and helps track patch location, epoch, and sampling mode.
       

        if self.return_meta:
            meta = {
                "case_dir": case_dir,
                "epoch": epoch,
                "case_idx": case_idx,
                "patch_j": patch_j,
                "cy": cy,
                "cx": cx,
                "fov": fov,     # 0=head, 1=body
                "strong": strong,
                "conditional": self.conditional,
            }
            out.update(meta)

        return out
