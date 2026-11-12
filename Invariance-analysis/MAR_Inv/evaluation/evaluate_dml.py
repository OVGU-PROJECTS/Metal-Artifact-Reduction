
"""Evaluate invariance banks with DML."""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from MAR_Inv.dml.network import MetricResNet
from MAR_Inv.evaluation.evaluate_vae_md import (
    bootstrap_mean_ci,
    describe,
    distribution_stats,
    hu_scale_for_patch,
    load_stats,
    metadata_value,
    parse_convergence_counts,
    save_csv,
)


def load_model(path: str, device: torch.device) -> MetricResNet:
    checkpoint = torch.load(path, map_location="cpu")
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise KeyError("Expected checkpoint keys 'config' and 'model_state_dict'.")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model = MetricResNet.from_args(Namespace(**checkpoint["config"]))
    model.load_state_dict(state)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode(
    model: MetricResNet,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Embed images in GPU-safe chunks and return embeddings on CPU."""
    parts: List[torch.Tensor] = []
    for start in range(0, int(images.shape[0]), int(batch_size)):
        batch = images[start : start + int(batch_size)].to(
            device, non_blocking=True
        ).float()
        parts.append(model(batch).cpu())
    if not parts:
        raise ValueError("Cannot encode an empty tensor.")
    return torch.cat(parts, dim=0)


def final_row(name: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_bank_name": name,
        "sample_bank": summary["sample_bank"],
        "num_patches": summary["num_patches"],
        "num_samples_per_patch": summary["num_samples_per_patch"],
        "num_patch_sample_pairs": summary["num_patch_sample_pairs"],
        "final_sdml": summary["final_sdml"],
        "sdml_patch_std": summary["sdml_std"],
        "sdml_patch_se": summary["sdml_mean_patch_se"],
        "sdml_ci_low": summary["sdml_mean_ci_low"],
        "sdml_ci_high": summary["sdml_mean_ci_high"],
        "confidence_level": summary["confidence_level"],
        "final_md_normalized": summary["final_md_normalized"],
        "final_md_hu": summary.get("final_md_hu", float("nan")),
    }


def convergence_rows(
    name: str,
    bank_path: str,
    cosine: np.ndarray,
    md: np.ndarray,
    hu_scales: np.ndarray,
    counts: Sequence[int],
    confidence: float,
    repetitions: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    max_k = int(cosine.shape[1])
    for requested in counts:
        k = min(int(requested), max_k)
        sdml = cosine[:, :k].mean(axis=1)
        md_norm = md[:, :k].mean(axis=1)
        md_hu = md_norm * hu_scales
        finite_hu = md_hu[np.isfinite(md_hu)]
        ci_low, ci_high = bootstrap_mean_ci(
            sdml,
            confidence=confidence,
            repetitions=repetitions,
            seed=seed + k,
        )
        rows.append(
            {
                "sample_bank_name": name,
                "sample_bank": os.path.abspath(bank_path),
                "requested_num_samples": int(requested),
                "used_num_samples": k,
                "num_patches": int(cosine.shape[0]),
                "num_patch_sample_pairs": int(cosine.shape[0] * k),
                "final_sdml": float(sdml.mean()),
                "sdml_patch_std": float(sdml.std(ddof=1)) if sdml.size > 1 else 0.0,
                "sdml_patch_se": (
                    float(sdml.std(ddof=1) / np.sqrt(sdml.size))
                    if sdml.size > 1 else 0.0
                ),
                "sdml_ci_low": ci_low,
                "sdml_ci_high": ci_high,
                "confidence_level": confidence,
                "final_md_normalized": float(md_norm.mean()),
                "final_md_hu": (
                    float(finite_hu.mean()) if finite_hu.size else float("nan")
                ),
            }
        )
    return rows


@torch.no_grad()
def evaluate_bank(
    model: MetricResNet,
    bank_path: str,
    *,
    device: torch.device,
    embedding_batch_size: int,
    patch_batch_size: int,
    max_items: int,
    stats_path: str,
    repetitions: int,
    confidence: float,
    seed: int,
    out_dir: Path,
    counts: Sequence[int],
) -> Dict[str, Any]:
    bank = torch.load(bank_path, map_location="cpu")
    missing = sorted({"x", "samples", "m"} - set(bank))
    if missing:
        raise KeyError(f"Sample bank is missing keys: {missing}")
    x, samples, mask = bank["x"].float(), bank["samples"].float(), bank["m"].float()
    if x.ndim != 4 or samples.ndim != 5:
        raise ValueError("Expected x [N,C,H,W] and samples [N,K,C,H,W].")
    if x.shape[0] != samples.shape[0] or x.shape[1:] != samples.shape[2:]:
        raise ValueError("Original and sample shapes are inconsistent.")
    if mask.shape != x.shape:
        raise ValueError("Expected mask m to have the same shape as x.")

    n = int(x.shape[0])
    if max_items > 0:
        n = min(n, int(max_items))
        x, samples, mask = x[:n], samples[:n], mask[:n]
    k = int(samples.shape[1])
    if n == 0 or k == 0:
        raise ValueError("The sample bank is empty.")

    cosine_parts: List[torch.Tensor] = []
    md_parts: List[torch.Tensor] = []
    for start in tqdm(
        range(0, n, patch_batch_size), desc=f"evaluate {Path(bank_path).stem}"
    ):
        stop = min(n, start + patch_batch_size)
        xb, sb = x[start:stop], samples[start:stop]
        b = int(xb.shape[0])
        x_embedding = encode(model, xb, device, embedding_batch_size)
        sample_embedding = encode(
            model,
            sb.reshape(b * k, *sb.shape[2:]),
            device,
            embedding_batch_size,
        ).reshape(b, k, -1)
        cosine_parts.append(
            F.cosine_similarity(x_embedding[:, None, :], sample_embedding, dim=2)
        )

        nonmetal = (mask[start:stop] <= 0.5).float()
        denominator = nonmetal.flatten(1).sum(1).clamp_min(1.0)
        difference = (sb - xb[:, None]).abs() * nonmetal[:, None]
        md_parts.append(difference.flatten(2).sum(2) / denominator[:, None])

    cosine = torch.cat(cosine_parts).numpy().astype(np.float64)
    md = torch.cat(md_parts).numpy().astype(np.float64)
    sdml, md_norm = cosine.mean(1), md.mean(1)
    meta = bank.get("meta", [])
    stats = load_stats(bank, stats_path)
    hu_scales = np.asarray(
        [hu_scale_for_patch(meta, index, stats) for index in range(n)],
        dtype=np.float64,
    )
    md_hu_per_sample = md * hu_scales[:, None]
    md_hu = md_norm * hu_scales
    name = Path(bank_path).stem

    per_patch: List[Dict[str, Any]] = []
    for index in range(n):
        row: Dict[str, Any] = {
            "patch_index": index,
            "case_dir": metadata_value(meta, index, "case_dir"),
            "fov": metadata_value(meta, index, "fov"),
            "strong": metadata_value(meta, index, "strong"),
            "num_samples": k,
        }
        row.update(distribution_stats(cosine[index], "sdml"))
        row.update(distribution_stats(md[index], "md_normalized"))
        row.update(distribution_stats(md_hu_per_sample[index], "md_hu"))
        per_patch.append(row)

    stat_names = [
        "mean", "std", "mc_se", "median", "min", "q05", "q25",
        "q75", "q95", "max", "iqr",
    ]
    fields = ["patch_index", "case_dir", "fov", "strong", "num_samples"]
    for prefix in ("sdml", "md_normalized", "md_hu"):
        fields.extend(f"{prefix}_{stat}" for stat in stat_names)
    save_csv(out_dir / f"{name}_per_patch_metrics.csv", per_patch, fields)

    ci_low, ci_high = bootstrap_mean_ci(
        sdml, confidence=confidence, repetitions=repetitions, seed=seed
    )
    summary: Dict[str, Any] = {
        "sample_bank": os.path.abspath(bank_path),
        "num_patches": n,
        "num_samples_per_patch": k,
        "num_patch_sample_pairs": n * k,
        "confidence_level": confidence,
        "bootstrap_repetitions": repetitions,
        "sdml_mean_ci_low": ci_low,
        "sdml_mean_ci_high": ci_high,
        "sdml_mean_patch_se": (
            float(sdml.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        ),
        "sdml_mean_within_patch_mc_se": (
            float(np.mean(cosine.std(1, ddof=1) / np.sqrt(k))) if k > 1 else 0.0
        ),
        **describe(sdml, "sdml"),
        **describe(md_norm, "md_normalized"),
    }
    summary["final_sdml"] = summary["sdml_mean"]
    summary["final_md_normalized"] = summary["md_normalized_mean"]
    finite_hu = md_hu[np.isfinite(md_hu)]
    if finite_hu.size:
        summary.update(describe(finite_hu, "md_hu"))
        summary["final_md_hu"] = summary["md_hu_mean"]
    else:
        summary["final_md_hu"] = float("nan")
        summary["md_hu_note"] = "HU scaling unavailable; provide valid stats and fov metadata."

    rows = convergence_rows(
        name, bank_path, cosine, md, hu_scales, counts,
        confidence, repetitions, seed,
    )
    summary["convergence_counts"] = [row["used_num_samples"] for row in rows]
    summary["convergence"] = rows
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample_bank", required=True, nargs="+")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--patch_batch_size", type=int, default=2)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--stats", default="")
    parser.add_argument("--bootstrap_repetitions", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1332)
    parser.add_argument(
        "--convergence_counts", default="1,2,5,10,20,50,100,200,500,1000"
    )
    args = parser.parse_args()
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between 0 and 1.")
    if args.embedding_batch_size <= 0 or args.patch_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap_repetitions must be positive.")

    counts = parse_convergence_counts(args.convergence_counts)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device)

    summaries: Dict[str, Any] = {}
    final_rows: List[Dict[str, Any]] = []
    all_convergence: List[Dict[str, Any]] = []
    for offset, bank_path in enumerate(args.sample_bank):
        name = Path(bank_path).stem
        summary = evaluate_bank(
            model, bank_path, device=device,
            embedding_batch_size=args.embedding_batch_size,
            patch_batch_size=args.patch_batch_size,
            max_items=args.max_items, stats_path=args.stats,
            repetitions=args.bootstrap_repetitions, confidence=args.confidence,
            seed=args.seed + offset, out_dir=out_dir, counts=counts,
        )
        summaries[name] = summary
        final_rows.append(final_row(name, summary))
        all_convergence.extend(summary["convergence"])
        print("\n" + "=" * 72)
        print(f"Final DML result: {name}")
        print(f"N patches              : {summary['num_patches']}")
        print(f"K samples per patch    : {summary['num_samples_per_patch']}")
        print(f"Final S_DML            : {summary['final_sdml']:.8f}")
        print(
            f"{round(100 * summary['confidence_level'])}% bootstrap CI       : "
            f"[{summary['sdml_mean_ci_low']:.8f}, {summary['sdml_mean_ci_high']:.8f}]"
        )
        print(f"Final MD normalized    : {summary['final_md_normalized']:.8f}")
        print(f"Final MD [HU]          : {summary['final_md_hu']:.8f}")
        print("=" * 72)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True, allow_nan=True)
    final_fields = [
        "sample_bank_name", "sample_bank", "num_patches", "num_samples_per_patch",
        "num_patch_sample_pairs", "final_sdml", "sdml_patch_std", "sdml_patch_se",
        "sdml_ci_low", "sdml_ci_high", "confidence_level", "final_md_normalized",
        "final_md_hu",
    ]
    convergence_fields = [
        "sample_bank_name", "sample_bank", "requested_num_samples", "used_num_samples",
        "num_patches", "num_patch_sample_pairs", "final_sdml", "sdml_patch_std",
        "sdml_patch_se", "sdml_ci_low", "sdml_ci_high", "confidence_level",
        "final_md_normalized", "final_md_hu",
    ]
    save_csv(out_dir / "final_dml_results.csv", final_rows, final_fields)
    save_csv(out_dir / "sample_count_convergence.csv", all_convergence, convergence_fields)
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
'''
 python -u -m MAR_Inv.evaluation.evaluate_dml \
  --checkpoint "results/runs/evaluation/dml/checkpoints/best.pt" \
  --sample_bank "results/sample_banks/test_banks/dudonet_head_full_k1000_seed2026.pt" \
  --out_dir "results/dml_new" \
  --device cuda:0 \
  --embedding_batch_size 32 \
  --patch_batch_size 1 \
  --max_items 0 \
  --stats configs/head_body_stats_testset.json \
  --bootstrap_repetitions 10000 \
  --confidence 0.95 \
  --convergence_counts 1,2,5,10,20,50,100,200,500,1000 \
  --seed 1332 

'''