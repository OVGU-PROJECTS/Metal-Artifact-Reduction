"""Staged Optuna finetuning for the unified PConv-MAR trainer"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import optuna
from optuna.storages import RDBStorage
from optuna.trial import TrialState

from .internal import launcher
from .internal.config import build_trial_config, load_json, write_json_atomic, write_trial_yaml
from .internal.scoring import best_validation


def _make_storage(study_dir: Path) -> RDBStorage:
    """SQLite study DB inside the study dir; reopening it is what resumes a stage."""
    return RDBStorage(
        url=f"sqlite:///{study_dir / 'optuna.db'}?journal_mode=WAL",
        engine_kwargs={"connect_args": {"timeout": 30, "check_same_thread": False, "isolation_level": None}},
    )


def _dry_run_score(sampled: dict) -> float:
    """Deterministic fake score, so --dry-run exercises the full Optuna loop."""
    payload = json.dumps(sampled, sort_keys=True, separators=(",", ":"))
    return float(sum(payload.encode("utf-8")) % 1000) / 1000.0


def suggest_value(name: str, spec: Mapping[str, Any], trial: optuna.Trial) -> Any:
    """Draw one parameter from its search-space spec."""
    kind = spec["type"]
    if kind == "choice":
        return trial.suggest_categorical(name, list(spec["values"]))
    if kind == "loguniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    raise ValueError(f"Unsupported search-space type for '{name}': {kind}")


def objective_stage(trial, args, base_config, stage_config, study_dir: Path) -> float:
    sampled = {name: suggest_value(name, spec, trial)
               for name, spec in stage_config["search_space"].items()}

    trial_dir = study_dir / f"trial_{trial.number:04d}"
    config = build_trial_config(base_config, stage_config, sampled, trial_dir)
    train_args = config["train_args"]

    write_json_atomic(trial_dir / "trial_config.json", config)
    write_trial_yaml(train_args, trial_dir)

    trial.set_user_attr("trial_dir", str(trial_dir))
    trial.set_user_attr("sampled_params", dict(sampled))

    rc = launcher.run_training(
        trial_dir,
        n_gpus=train_args["n_gpus"],
        log_path=trial_dir / "launch.log",
        dry_run=args.dry_run,
    )
    if rc != 0:
        raise optuna.TrialPruned(f"training failed (exit {rc}) in {trial_dir}")

    if args.dry_run:
        result = {"score": _dry_run_score(sampled), "best_epoch": None, "metrics": {}, "reason": "dry_run"}
    else:
        result = best_validation(trial_dir, config["score"])

    write_json_atomic(trial_dir / "result.json", {"trial_number": trial.number, **result})
    if result["score"] is None:
        raise optuna.TrialPruned(f"no usable score ({result['reason']}) in {trial_dir}")

    trial.set_user_attr("best_epoch", result["best_epoch"])
    trial.set_user_attr("metrics", result["metrics"])
    trial.report(result["score"], step=train_args["epochs"])
    return result["score"]


def write_ranking_json(study, study_dir: Path) -> None:
    ranked = sorted((t for t in study.trials if t.state == TrialState.COMPLETE),
                    key=lambda t: t.value)
    if not ranked:
        print("[finetune] no completed trials to rank.", flush=True)
        return

    best = ranked[0]
    write_json_atomic(study_dir / "ranking.json", {
        "study_name": study.study_name,
        "direction": "minimize",
        "n_trials_total": len(study.trials),
        "n_trials_completed": len(ranked),
        "best_trial_number": best.number,
        "best_score": best.value,
        "best_params": best.user_attrs["sampled_params"],
        "ranking_best_to_worst": [{
            "rank": i,
            "trial_number": t.number,
            "score": t.value,
            "best_epoch": t.user_attrs["best_epoch"],
            "trial_dir": t.user_attrs["trial_dir"],
            "sampled_params": t.user_attrs["sampled_params"],
        } for i, t in enumerate(ranked, start=1)],
    })

    # Promote the best trial's frozen config for the next stage's --base.
    best_config = load_json(Path(best.user_attrs["trial_dir"]) / "trial_config.json")
    write_json_atomic(study_dir / "best_config.json", best_config)

    print(f"\n[finetune] best trial #{best.number} score={best.value:.6f}", flush=True)
    print(f"[finetune] ranking + best_config.json written to {study_dir}", flush=True)


def run_study(args) -> optuna.Study:
    study_dir = Path(args.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_json(args.base)
    stage_config = load_json(args.stage)

    study = optuna.create_study(
        study_name=Path(args.stage).stem,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, constant_liar=True),
        storage=_make_storage(study_dir),
        load_if_exists=True,
    )

    # --trials 0 re-emits ranking.json from an existing study without training.
    if args.trials > 0:
        study.optimize(
            lambda trial: objective_stage(trial, args, base_config, stage_config, study_dir),
            n_trials=args.trials,
            gc_after_trial=True,
        )

    write_ranking_json(study, study_dir)
    return study
