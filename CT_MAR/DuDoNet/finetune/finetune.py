"""The thesis DuDoNet learning-rate search: 50 trials of 12 epochs by default."""
import os
import argparse
import optuna
import json
from pathlib import Path
from optuna.trial import TrialState
from optuna.samplers import TPESampler

def _is_rank0() -> bool:
    # only rank0 writes files.
    return os.environ.get("RANK", "0") == "0" and os.environ.get("LOCAL_RANK", "0") == "0"


def _lr_tag(lr: float) -> str:
    # folder-friendly scientific notation, avoids '+' in names
    return f"{lr:.2e}".replace("+", "")


def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)

def objective_phase1(trial, args):
    """One Optuna trial: 
    1- sample an LR
    2- run a short proxy train, 
    - Optuna minimizes the val loss
    - Only LR is searched; betas are fixed to the paper values.
    """
    
    # Help and ranking-only invocations do not need the CUDA training stack.
    from training.trainer import train_dudonet

    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)  # log-uniform
    beta1, beta2 = 0.5, 0.999  # paper values, fixed
    trial_dir = Path(args.out_dir) / f"trial_{trial.number:05d}_lr{_lr_tag(lr)}"

    if _is_rank0():
        # store metadata in optuna
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("beta1", float(beta1))
        trial.set_user_attr("beta2", float(beta2))

        # write config.json
        write_json_atomic(trial_dir / "config.json", {
            "trial_number": trial.number,
            "lr": float(lr),
            "beta1": float(beta1),
            "beta2": float(beta2),
        })
    best_val = train_dudonet(
        train_root=args.train_root,
        val_root=args.val_root,
        out_dir=str(trial_dir),  
        fov_head_mm=args.fov_head_mm,
        fov_body_mm=args.fov_body_mm,
        batch_size=args.batch_size,
        epochs=args.epochs,                
        max_epochs_for_smbo=args.smbo_epochs,
        trainset=args.trainset,
        valset=args.valset,
        seed=args.seed,
        lr=lr,
        beta1=beta1, beta2=beta2,
        accum_steps=args.accum_steps,
    )
    
    if _is_rank0():
        # write result.json
        write_json_atomic(trial_dir / "result.json", {
            "trial_number": trial.number,
            "best_val": float(best_val),
        })

    return float(best_val)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", type=str, default="sqlite:////mnt/public/bahaa/smbo/dudonet_smbo.db")
    ap.add_argument("--study", type=str, default="dudonet_phase1")

    ap.add_argument("--train_root", type=str, required=True)
    ap.add_argument("--val_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default= "/mnt/public/bahaa/smbo/dudonet_phase1")

    ap.add_argument("--fov_head_mm", type=float, default=220.16)
    ap.add_argument("--fov_body_mm", type=float, default=400.0)

    ap.add_argument("--batch_size", type=int, default=3)
    ap.add_argument("--accum_steps", type=int, default=None)

    ap.add_argument("--epochs", type=int, default=380)     # paper default
    ap.add_argument("--smbo_epochs", type=int, default=12) # short proxy run
    ap.add_argument("--trainset", type=int, default=1000)   #fixed proxy subset
    ap.add_argument("--valset", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--trials", type=int, default=50)

    args = ap.parse_args()

    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        sampler=TPESampler(constant_liar=True),

    )
    study.optimize(lambda t: objective_phase1(t, args), n_trials=args.trials)
    
    def write_ranking_json(study: optuna.Study, smbo_dir: str) -> None:
        smbo_dir = Path(smbo_dir)
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]  
        completed.sort(key=lambda t: float(t.value))

        ranking = []
        for rank, t in enumerate(completed, start=1):
            ranking.append({
                "rank": rank,
                "trial_number": t.number,
                "best_val": float(t.value),
                "config": {
                    "lr": float(t.params.get("lr")) if "lr" in t.params else None,
                    "beta1": float(t.user_attrs.get("beta1", 0.5)),
                    "beta2": float(t.user_attrs.get("beta2", 0.999)),
                },
                "trial_dir": t.user_attrs.get("trial_dir"),
            })

        write_json_atomic(smbo_dir / "ranking.json", {
            "study_name": study.study_name,
            "direction": "minimize",
            "n_trials_total": len(study.trials),
            "n_trials_complete": len(completed),
            "ranking_best_to_worst": ranking,
        })

    # --- after study.optimize(...) ---
    if _is_rank0():
        write_ranking_json(study, args.out_dir)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)




if __name__ == "__main__":
    main()

