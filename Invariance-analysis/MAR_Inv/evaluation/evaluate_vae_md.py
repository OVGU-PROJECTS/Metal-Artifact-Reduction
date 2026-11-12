#!/usr/bin/env python3
"""Evaluate invariance sample banks with an unconditional VAE.

For every original corrupted patch x_i and each of its K invariance samples
x_tilde_i,k, this script computes the per-sample cosine similarity

    s_VAE(i, k) = cosine(mu(x_i), mu(x_tilde_i,k)),

then the patch-wise score

    S_VAE(i) = (1 / K) * sum_k s_VAE(i, k),

and finally the model-level score

    S_VAE = (1 / N) * sum_i S_VAE(i).

The posterior mean is used by default to keep the evaluator deterministic and
to prevent VAE reparameterization noise from entering the similarity score.
The script also reports the non-metal mean absolute difference (MD) and a
bootstrap confidence interval for the final patch-averaged score.

Outputs
-------
For every sample bank <name>.pt:
    <name>_per_patch_metrics.csv

Each CSV row corresponds to one original patch. The K generated samples are
summarized by distribution statistics (mean, standard deviation, Monte Carlo
standard error, median, minimum, 5th/25th/75th/95th percentiles, maximum, and
interquartile range) for cosine similarity and non-metal MD.

For the complete run:
    final_unconditional_vae_results.csv
    sample_count_convergence.csv
    summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from MAR_Inv.vae.network import BigAE


def _clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove a possible DataParallel 'module.' prefix from checkpoint keys."""
    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def load_unconditional_vae(checkpoint_path: str, device: torch.device) -> BigAE:
    """Load and freeze the unconditional VAE used only as an encoder."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in checkpoint or "generator_state_dict" not in checkpoint:
        raise KeyError("Expected checkpoint keys 'config' and 'generator_state_dict'.")

    config = checkpoint["config"]
    if bool(config.get("conditional", False)):
        raise ValueError(
            "The evaluation checkpoint must belong to an unconditional VAE "
            "with conditional: false."
        )

    model = BigAE(Namespace(**config), in_ch=1, out_ch=1, cond_ch=0)
    model.load_state_dict(_clean_state_dict(checkpoint["generator_state_dict"]))
    model.eval()

    # Only the encoder is needed. Leaving the decoder on CPU reduces GPU use.
    model.encoder.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode(
    model: BigAE,
    images: torch.Tensor,
    *,
    mode: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Encode images into flattened unconditional-VAE embeddings."""
    embeddings: List[torch.Tensor] = []
    for start in range(0, int(images.shape[0]), int(batch_size)):
        batch = images[start : start + int(batch_size)].to(
            device, non_blocking=True
        ).float()
        posterior = model.encode(batch)
        latent = posterior.mode() if mode == "mean" else posterior.sample()
        embeddings.append(latent.flatten(1).cpu())

    if not embeddings:
        raise ValueError("Cannot encode an empty image tensor.")
    return torch.cat(embeddings, dim=0)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float,
    repetitions: int,
    seed: int,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean across test patches."""
    if values.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(int(repetitions), dtype=np.float64)
    for start in range(0, int(repetitions), 1000):
        count = min(1000, int(repetitions) - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        bootstrap_means[start : start + count] = values[indices].mean(axis=1)

    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def describe(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """Return standard descriptive statistics for a non-empty array."""
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_q25": float("nan"),
            f"{prefix}_q75": float("nan"),
            f"{prefix}_max": float("nan"),
        }

    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_max": float(values.max()),
    }


def distribution_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """Describe the K sample-level values belonging to one patch."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_mc_se": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_q05": float("nan"),
            f"{prefix}_q25": float("nan"),
            f"{prefix}_q75": float("nan"),
            f"{prefix}_q95": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_iqr": float("nan"),
        }

    q05, q25, q75, q95 = np.quantile(finite, [0.05, 0.25, 0.75, 0.95])
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_std": std,
        f"{prefix}_mc_se": std / float(np.sqrt(finite.size)) if finite.size > 1 else 0.0,
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_min": float(finite.min()),
        f"{prefix}_q05": float(q05),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_q95": float(q95),
        f"{prefix}_max": float(finite.max()),
        f"{prefix}_iqr": float(q75 - q25),
    }


def metadata_value(
    meta: Sequence[Dict[str, Any]], index: int, key: str, default: Any = ""
) -> Any:
    """Safely retrieve one metadata value from a sample-bank metadata list."""
    if index >= len(meta) or not isinstance(meta[index], dict):
        return default
    value = meta[index].get(key, default)
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else value.tolist()
    return value


def load_stats(
    bank: Dict[str, Any], explicit_path: str
) -> Optional[Dict[str, Any]]:
    """Load normalization statistics from CLI or sample-bank configuration."""
    stats_path = explicit_path
    if not stats_path:
        stats_path = str(bank.get("config", {}).get("stats", ""))
    if not stats_path or not os.path.isfile(stats_path):
        return None
    with open(stats_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def hu_scale_for_patch(
    meta: Sequence[Dict[str, Any]],
    index: int,
    stats: Optional[Dict[str, Any]],
) -> float:
    """Return the z-score standard deviation needed to convert differences to HU."""
    if stats is None:
        return float("nan")

    fov = metadata_value(meta, index, "fov", default=None)
    if fov is None:
        return float("nan")

    group = "head" if int(fov) == 0 else "body"
    try:
        return float(stats[group]["std"])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def save_csv(
    path: Path,
    rows: Iterable[Dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Write dictionaries to CSV using a fixed field order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def final_result_row(bank_name: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Create the compact one-row final result for one evaluated sample bank."""
    return {
        "sample_bank_name": bank_name,
        "sample_bank": summary["sample_bank"],
        "embedding_mode": summary["embedding_mode"],
        "num_patches": summary["num_patches"],
        "num_samples_per_patch": summary["num_samples_per_patch"],
        "num_patch_sample_pairs": summary["num_patch_sample_pairs"],
        # This is Equation 4.56: mean of patch-wise S_VAE(i).
        "final_svae": summary["final_svae"],
        "svae_patch_std": summary["svae_std"],
        "svae_patch_se": summary["svae_mean_patch_se"],
        "svae_ci_low": summary["svae_mean_ci_low"],
        "svae_ci_high": summary["svae_mean_ci_high"],
        "confidence_level": summary["confidence_level"],
        "final_md_normalized": summary["final_md_normalized"],
        "final_md_hu": summary.get("final_md_hu", float("nan")),
    }


def parse_convergence_counts(text: str) -> List[int]:
    """Parse comma-separated sample counts used for K-convergence reporting."""
    counts: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("--convergence_counts must contain positive integers.")
        counts.append(value)
    if not counts:
        raise ValueError("--convergence_counts cannot be empty.")
    return sorted(set(counts))


def convergence_rows(
    *,
    bank_name: str,
    bank_path: str,
    cosine: np.ndarray,
    md_normalized: np.ndarray,
    hu_scales: np.ndarray,
    counts: Sequence[int],
    confidence: float,
    bootstrap_repetitions: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Summarize final metrics when only the first K samples are used."""
    rows: List[Dict[str, Any]] = []
    max_k = int(cosine.shape[1])
    for requested_count in counts:
        k_used = min(int(requested_count), max_k)
        svae_k = cosine[:, :k_used].mean(axis=1)
        md_norm_k = md_normalized[:, :k_used].mean(axis=1)
        md_hu_k = md_norm_k * hu_scales
        finite_hu = md_hu_k[np.isfinite(md_hu_k)]
        ci_low, ci_high = bootstrap_mean_ci(
            svae_k,
            confidence=confidence,
            repetitions=bootstrap_repetitions,
            seed=seed + k_used,
        )
        rows.append(
            {
                "sample_bank_name": bank_name,
                "sample_bank": os.path.abspath(bank_path),
                "requested_num_samples": int(requested_count),
                "used_num_samples": k_used,
                "num_patches": int(cosine.shape[0]),
                "num_patch_sample_pairs": int(cosine.shape[0] * k_used),
                "final_svae": float(svae_k.mean()),
                "svae_patch_std": float(svae_k.std(ddof=1)) if svae_k.size > 1 else 0.0,
                "svae_patch_se": (
                    float(svae_k.std(ddof=1) / np.sqrt(svae_k.size))
                    if svae_k.size > 1
                    else 0.0
                ),
                "svae_ci_low": ci_low,
                "svae_ci_high": ci_high,
                "confidence_level": confidence,
                "final_md_normalized": float(md_norm_k.mean()),
                "final_md_hu": float(finite_hu.mean()) if finite_hu.size else float("nan"),
            }
        )
    return rows


@torch.no_grad()
def evaluate_bank(
    model: BigAE,
    bank_path: str,
    *,
    device: torch.device,
    embedding_mode: str,
    embedding_batch_size: int,
    patch_batch_size: int,
    max_items: int,
    stats_path: str,
    bootstrap_repetitions: int,
    confidence: float,
    seed: int,
    out_dir: Path,
    convergence_counts: Sequence[int],
) -> Dict[str, Any]:
    """Evaluate one sample bank and write one row per patch."""
    bank = torch.load(bank_path, map_location="cpu")
    required = {"x", "samples"}
    missing = sorted(required - set(bank))
    if missing:
        raise KeyError(f"Sample bank is missing keys: {missing}")

    x = bank["x"].float()
    samples = bank["samples"].float()
    if x.ndim != 4 or samples.ndim != 5:
        raise ValueError("Expected x [N,C,H,W] and samples [N,K,C,H,W].")
    if x.shape[0] != samples.shape[0] or x.shape[1:] != samples.shape[2:]:
        raise ValueError("The original and sampled patch shapes are inconsistent.")

    n = int(x.shape[0])
    if int(max_items) > 0:
        n = min(n, int(max_items))
        x = x[:n]
        samples = samples[:n]

    k = int(samples.shape[1])
    if n == 0 or k == 0:
        raise ValueError("The sample bank is empty.")

    mask = bank.get("m")
    if mask is None:
        raise KeyError(
            "Sample bank has no metal mask 'm'; non-metal MD cannot be computed."
        )
    mask = mask[:n].float()
    if mask.shape != x.shape:
        raise ValueError("Expected mask m to have the same shape as x.")

    meta = bank.get("meta", [])
    stats = load_stats(bank, stats_path)

    cosine_rows: List[torch.Tensor] = []
    md_rows: List[torch.Tensor] = []

    for start in tqdm(
        range(0, n, int(patch_batch_size)),
        desc=f"evaluate {Path(bank_path).stem}",
    ):
        stop = min(n, start + int(patch_batch_size))
        x_batch = x[start:stop]
        sample_batch = samples[start:stop]
        b = int(x_batch.shape[0])

        # Original embedding: [B,D]
        x_embedding = encode(
            model,
            x_batch,
            mode=embedding_mode,
            device=device,
            batch_size=embedding_batch_size,
        )

        # Sample embeddings: [B,K,D]
        sample_embedding = encode(
            model,
            sample_batch.reshape(b * k, *sample_batch.shape[2:]),
            mode=embedding_mode,
            device=device,
            batch_size=embedding_batch_size,
        ).reshape(b, k, -1)

        # Unconditional-VAE similarity for all K samples of each patch: [B,K]
        cosine_rows.append(
            F.cosine_similarity(
                x_embedding[:, None, :],
                sample_embedding,
                dim=2,
            ).cpu()
        )

        # Non-metal MAE for all K samples of each patch: [B,K]
        nonmetal = (mask[start:stop] <= 0.5).float()
        denominator = nonmetal.flatten(1).sum(dim=1).clamp_min(1.0)
        absolute_difference = (
            (sample_batch - x_batch[:, None]).abs() * nonmetal[:, None]
        )
        md_rows.append(
            absolute_difference.flatten(2).sum(dim=2) / denominator[:, None]
        )

    # [N,K]
    cosine = torch.cat(cosine_rows, dim=0).numpy().astype(np.float64)
    md_normalized = torch.cat(md_rows, dim=0).numpy().astype(np.float64)

    # Patch-wise results.
    svae = cosine.mean(axis=1)
    md_norm = md_normalized.mean(axis=1)

    # One HU scale per patch. A normalized difference is converted by multiplying
    # with the same standard deviation used during z-score normalization.
    hu_scales = np.asarray(
        [hu_scale_for_patch(meta, index, stats) for index in range(n)],
        dtype=np.float64,
    )
    md_hu_per_sample = md_normalized * hu_scales[:, None]
    md_hu_values = md_norm * hu_scales

    name = Path(bank_path).stem

    # ------------------------------------------------------------------
    # Per-patch CSV: one row per original patch, summarizing all K samples
    # ------------------------------------------------------------------
    per_patch_rows: List[Dict[str, Any]] = []
    for index in range(n):
        row: Dict[str, Any] = {
            "patch_index": index,
            "case_dir": metadata_value(meta, index, "case_dir"),
            "fov": metadata_value(meta, index, "fov"),
            "strong": metadata_value(meta, index, "strong"),
            "num_samples": k,
        }

        # Distribution of cosine similarity over the K samples for this patch.
        # svae_mean is the documented patch-wise S_VAE(i).
        row.update(distribution_stats(cosine[index], "svae"))

        # Distribution of non-metal mean absolute differences in normalized units.
        row.update(distribution_stats(md_normalized[index], "md_normalized"))

        # The same MD distribution converted back to HU when statistics
        # are available. Otherwise, these fields are written as NaN.
        row.update(distribution_stats(md_hu_per_sample[index], "md_hu"))
        per_patch_rows.append(row)

    per_patch_fields = [
        "patch_index",
        "case_dir",
        "fov",
        "strong",
        "num_samples",
        "svae_mean",
        "svae_std",
        "svae_mc_se",
        "svae_median",
        "svae_min",
        "svae_q05",
        "svae_q25",
        "svae_q75",
        "svae_q95",
        "svae_max",
        "svae_iqr",
        "md_normalized_mean",
        "md_normalized_std",
        "md_normalized_mc_se",
        "md_normalized_median",
        "md_normalized_min",
        "md_normalized_q05",
        "md_normalized_q25",
        "md_normalized_q75",
        "md_normalized_q95",
        "md_normalized_max",
        "md_normalized_iqr",
        "md_hu_mean",
        "md_hu_std",
        "md_hu_mc_se",
        "md_hu_median",
        "md_hu_min",
        "md_hu_q05",
        "md_hu_q25",
        "md_hu_q75",
        "md_hu_q95",
        "md_hu_max",
        "md_hu_iqr",
    ]
    save_csv(
        out_dir / f"{name}_per_patch_metrics.csv",
        per_patch_rows,
        per_patch_fields,
    )

    # ----------------------------------------------------
    # Final unconditional-VAE score for the full test bank
    # --------------------------------------------------------------
    ci_low, ci_high = bootstrap_mean_ci(
        svae,
        confidence=confidence,
        repetitions=bootstrap_repetitions,
        seed=seed,
    )

    summary: Dict[str, Any] = {
        "sample_bank": os.path.abspath(bank_path),
        "embedding_mode": embedding_mode,
        "num_patches": n,
        "num_samples_per_patch": k,
        "num_patch_sample_pairs": n * k,
        "confidence_level": confidence,
        "bootstrap_repetitions": bootstrap_repetitions,
        "svae_mean_ci_low": ci_low,
        "svae_mean_ci_high": ci_high,
        "svae_mean_patch_se": (
            float(svae.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        ),
        "svae_mean_within_patch_mc_se": (
            float(np.mean(cosine.std(axis=1, ddof=1) / np.sqrt(k)))
            if k > 1
            else 0.0
        ),
        **describe(svae, "svae"),
        **describe(md_norm, "md_normalized"),
    }

    # Explicit aliases used in the final results CSV.
    summary["final_svae"] = summary["svae_mean"]
    summary["final_md_normalized"] = summary["md_normalized_mean"]

    finite_hu = md_hu_values[np.isfinite(md_hu_values)]
    if finite_hu.size:
        summary.update(describe(finite_hu, "md_hu"))
        summary["final_md_hu"] = summary["md_hu_mean"]
    else:
        summary["final_md_hu"] = float("nan")
        summary["md_hu_note"] = (
            "HU scaling unavailable; provide --stats or store a valid stats path "
            "in the sample-bank config and include fov metadata."
        )

    rows = convergence_rows(
        bank_name=name,
        bank_path=bank_path,
        cosine=cosine,
        md_normalized=md_normalized,
        hu_scales=hu_scales,
        counts=convergence_counts,
        confidence=confidence,
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )
    summary["convergence_counts"] = [row["used_num_samples"] for row in rows]
    summary["convergence"] = rows

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate invariance sample banks using an unconditional-VAE "
            "embedding and save per-patch and final CSV results."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Unconditional VAE checkpoint with conditional: false.",
    )
    parser.add_argument(
        "--sample_bank",
        required=True,
        nargs="+",
        help="One or more sample-bank .pt files.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--embedding_mode",
        choices=["mean", "sample"],
        default="mean",
        help="Use posterior mean (recommended) or stochastic latent sample.",
    )
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--patch_batch_size", type=int, default=4)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument(
        "--stats",
        default="",
        help="Optional normalization-statistics JSON override.",
    )
    parser.add_argument("--bootstrap_repetitions", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1332)
    parser.add_argument(
        "--convergence_counts",
        default="1,2,5,10,20,50,100,200,500,1000",
        help="Comma-separated K values for sample-count convergence reporting.",
    )
    args = parser.parse_args()

    if not (0.0 < args.confidence < 1.0):
        raise ValueError("--confidence must be between 0 and 1.")
    if args.embedding_batch_size <= 0 or args.patch_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap_repetitions must be positive.")
    convergence_counts = parse_convergence_counts(args.convergence_counts)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_unconditional_vae(args.checkpoint, device)

    summaries: Dict[str, Any] = {}
    final_rows: List[Dict[str, Any]] = []
    convergence_all_rows: List[Dict[str, Any]] = []

    for offset, bank_path in enumerate(args.sample_bank):
        name = Path(bank_path).stem
        summary = evaluate_bank(
            model,
            bank_path,
            device=device,
            embedding_mode=args.embedding_mode,
            embedding_batch_size=args.embedding_batch_size,
            patch_batch_size=args.patch_batch_size,
            max_items=args.max_items,
            stats_path=args.stats,
            bootstrap_repetitions=args.bootstrap_repetitions,
            confidence=args.confidence,
            seed=args.seed + offset,
            out_dir=out_dir,
            convergence_counts=convergence_counts,
        )
        summaries[name] = summary
        final_rows.append(final_result_row(name, summary))
        convergence_all_rows.extend(summary["convergence"])

        print("\n" + "=" * 72)
        print(f"Final unconditional-VAE result: {name}")
        print(f"N patches              : {summary['num_patches']}")
        print(f"K samples per patch    : {summary['num_samples_per_patch']}")
        print(f"Final S_VAE            : {summary['final_svae']:.8f}")
        print(
            f"{int(round(100 * summary['confidence_level']))}% bootstrap CI       : "
            f"[{summary['svae_mean_ci_low']:.8f}, "
            f"{summary['svae_mean_ci_high']:.8f}]"
        )
        print(f"Final MD normalized    : {summary['final_md_normalized']:.8f}")
        if np.isfinite(summary.get("final_md_hu", float("nan"))):
            print(f"Final MD [HU]          : {summary['final_md_hu']:.8f}")
        else:
            print("Final MD [HU]          : unavailable")
        print("=" * 72)

    # Full JSON with all statistics.
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True, allow_nan=True)

    # Compact final result table: one row per evaluated MAR sample bank.
    final_fields = [
        "sample_bank_name",
        "sample_bank",
        "embedding_mode",
        "num_patches",
        "num_samples_per_patch",
        "num_patch_sample_pairs",
        "final_svae",
        "svae_patch_std",
        "svae_patch_se",
        "svae_ci_low",
        "svae_ci_high",
        "confidence_level",
        "final_md_normalized",
        "final_md_hu",
    ]
    save_csv(
        out_dir / "final_unconditional_vae_results.csv",
        final_rows,
        final_fields,
    )

    convergence_fields = [
        "sample_bank_name",
        "sample_bank",
        "requested_num_samples",
        "used_num_samples",
        "num_patches",
        "num_patch_sample_pairs",
        "final_svae",
        "svae_patch_std",
        "svae_patch_se",
        "svae_ci_low",
        "svae_ci_high",
        "confidence_level",
        "final_md_normalized",
        "final_md_hu",
    ]
    save_csv(
        out_dir / "sample_count_convergence.csv",
        convergence_all_rows,
        convergence_fields,
    )

    print(f"\nSaved results to: {out_dir.resolve()}")
    print("  - final_unconditional_vae_results.csv")
    print("  - sample_count_convergence.csv")
    print("  - summary.json")
    for bank_path in args.sample_bank:
        name = Path(bank_path).stem
        print(f"  - {name}_per_patch_metrics.csv")


if __name__ == "__main__":
    main()
'''

python -u -m MAR_Inv.evaluation.evaluate_vae_md \
  --checkpoint "results/runs/evaluation/uncond_vae/checkpoints/best.pt" \
  --sample_bank "results/sample_banks/test_banks/dudonet_head_full_k1000_seed2026.pt" \
  --out_dir "results/vae_md_new" \
  --device cuda:0 \
  --embedding_mode mean \
  --embedding_batch_size 32 \
  --patch_batch_size 1 \
  --max_items 0 \
  --stats configs/head_body_stats_testset.json \
  --bootstrap_repetitions 10000 \
  --confidence 0.95 \
  --convergence_counts 1,2,5,10,20,50,100,200,500,1000 \
  --seed 1332 
  
  '''