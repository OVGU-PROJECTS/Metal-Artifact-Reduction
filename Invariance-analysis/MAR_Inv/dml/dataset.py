"""Datasets for DML training from precomputed invariance sample banks."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset


def load_sample_bank(path: str) -> Dict[str, Any]:
    bank = torch.load(path, map_location="cpu")
    required = {"x", "samples"}
    missing = sorted(required - set(bank))
    if missing:
        raise KeyError(f"Sample bank is missing required keys: {missing}")
    if bank["samples"].ndim != 5:
        raise ValueError("Expected bank['samples'] with shape [N,K,C,H,W].")
    if bank["x"].ndim != 4:
        raise ValueError("Expected bank['x'] with shape [N,C,H,W].")
    if bank["x"].shape[0] != bank["samples"].shape[0]:
        raise ValueError("bank['x'] and bank['samples'] must have the same N.")
    return bank


class DMLSampleBankDataset(Dataset):
    """One anchor and one same-case positive per item.

    Negatives are selected inside the batch by comparing every anchor against
    positives from other indices. This supports semi-hard triplet mining without
    hard-coding a single negative in the dataset.
    """

    def __init__(
        self,
        sample_bank_path: str,
        *,
        max_items: int = 0,
        positive_index: Optional[int] = None,
        samples_per_case: int = 0,
        randomize_samples: bool = False,
    ) -> None:
        super().__init__()
        self.bank = load_sample_bank(sample_bank_path)
        self.x = self.bank["x"].float()
        self.samples = self.bank["samples"].float()
        self.max_items = int(max_items)
        self.positive_index = positive_index
        self.samples_per_case = int(samples_per_case)
        self.randomize_samples = bool(randomize_samples)

        n = int(self.x.shape[0])
        if self.max_items > 0:
            n = min(n, self.max_items)
        self.n = n
        self.k = int(self.samples.shape[1])
        if self.samples_per_case > 0:
            self.samples_per_case = min(self.samples_per_case, self.k)

    def __len__(self) -> int:
        if self.samples_per_case > 0 and self.positive_index is None:
            return self.n * self.samples_per_case
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.samples_per_case > 0 and self.positive_index is None:
            case_idx = int(idx) // self.samples_per_case
            if self.randomize_samples:
                k = int(torch.randint(low=0, high=self.k, size=(1,)).item())
            else:
                k = int(idx) % self.samples_per_case
        elif self.positive_index is None:
            case_idx = int(idx)
            k = int(torch.randint(low=0, high=self.k, size=(1,)).item())
        else:
            case_idx = int(idx)
            k = int(self.positive_index) % self.k
        return {
            "anchor": self.x[case_idx],
            "positive": self.samples[case_idx, k],
            "label": torch.tensor(case_idx, dtype=torch.long),
            "sample_index": torch.tensor(k, dtype=torch.long),
        }


class DMLEvalSampleBankDataset(Dataset):
    """Evaluation view returning one anchor and all K invariance samples."""

    def __init__(self, sample_bank_path: str, *, max_items: int = 0) -> None:
        super().__init__()
        self.bank = load_sample_bank(sample_bank_path)
        self.x = self.bank["x"].float()
        self.samples = self.bank["samples"].float()
        n = int(self.x.shape[0])
        if int(max_items) > 0:
            n = min(n, int(max_items))
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.x[idx],
            "samples": self.samples[idx],
            "index": torch.tensor(idx, dtype=torch.long),
        }
