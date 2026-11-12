"""Score a finished trial from logs/loss_breakdown_val.csv."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping


def as_float(row: Mapping[str, str], key: str) -> float:
    """CSV cell as a float, NaN when the column is absent or not numeric"""
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def row_score(row: Mapping[str, str], score_config: Mapping[str, Any]) -> float:
    """Weighted sum of one breakdown row, lower is better. inf if any term is missing
    """
    ssim = as_float(row, score_config["ssim_metric"])

    total = 0.0
    for key, weight in score_config["weights"].items():
        value = 1.0 - ssim if key == "one_minus_ssim" else as_float(row, key)
        if not math.isfinite(value):
            return float("inf")
        total += weight * value
    return total


def best_validation(run_dir: str | Path, score_config: Mapping[str, Any]) -> dict[str, Any]:
    """Best-scoring epoch of a trial. "score" is None when no epoch is usable"""
    path = Path(run_dir) / "logs" / "loss_breakdown_val.csv"
    empty = {"score": None, "best_epoch": None, "metrics": {}}

    if not path.exists():
        return {**empty, "reason": "missing_breakdown"}

    with open(path, "r", encoding="utf-8") as f:
        scored = [(row_score(row, score_config), row) for row in csv.DictReader(f)]
    if not scored:
        return {**empty, "reason": "missing_breakdown"}

    score, row = min(scored, key=lambda item: item[0])
    if not math.isfinite(score):
        return {**empty, "reason": "no_finite_score"}

    return {
        "score": score,
        "best_epoch": int(float(row["epoch"])),
        "metrics": {key: as_float(row, key) for key in row if key != "epoch"},
        "reason": "",
    }
