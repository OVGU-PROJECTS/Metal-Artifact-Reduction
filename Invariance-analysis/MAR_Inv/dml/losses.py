"""Triplet losses and similarity helpers for DML evaluation."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    return 1.0 - a @ b.t()


def semihard_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    *,
    margin: float = 0.2,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Batch semi-hard triplet loss.

    For anchor i, the positive is positive i. Negatives are positive j, j != i.
    A semi-hard negative is farther than the positive but still violates the
    margin. If no semi-hard negative exists, the closest available negative is
    used so the batch remains informative.
    """

    if anchor.shape != positive.shape:
        raise ValueError("anchor and positive embeddings must have the same shape.")
    batch_size = int(anchor.shape[0])
    if batch_size < 2:
        raise ValueError("DML batch size must be at least 2 for in-batch negatives.")

    dist = cosine_distance(anchor, positive)
    pos_dist = dist.diagonal()

    if labels is None:
        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=dist.device)
    else:
        labels = labels.to(dist.device).view(-1)
        neg_mask = labels[:, None] != labels[None, :]
    neg_dist = dist.masked_fill(~neg_mask, float("inf"))
    if not torch.isfinite(neg_dist).any(dim=1).all():
        raise ValueError("Each anchor needs at least one different-label negative in the batch.")

    semi_mask = (neg_dist > pos_dist[:, None]) & (neg_dist < (pos_dist[:, None] + float(margin)))
    semi_values = neg_dist.masked_fill(~semi_mask, float("inf"))
    semi_neg = semi_values.min(dim=1).values

    closest_neg = neg_dist.min(dim=1).values
    chosen_neg = torch.where(torch.isfinite(semi_neg), semi_neg, closest_neg)

    loss_vec = F.relu(pos_dist - chosen_neg + float(margin))
    loss = loss_vec.mean()
    stats = {
        "pos_dist": pos_dist.mean().detach(),
        "neg_dist": chosen_neg.mean().detach(),
        "active_frac": (loss_vec > 0).float().mean().detach(),
        "semihard_frac": torch.isfinite(semi_neg).float().mean().detach(),
    }
    return loss, stats
