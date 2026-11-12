"""CLI entry point for one Optuna stage study"""
import argparse
from pathlib import Path
from .study import run_study

_DEFAULT_BASE = str(Path(__file__).resolve().parent / "configs" / "base.json")



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Optuna TPE study")
    ap.add_argument("--base", default=_DEFAULT_BASE, help="base config JSON (paths, calib, fixed args).")
    ap.add_argument("--stage", required=True, help="Stage JSON (fixed_args + search_space).")
    ap.add_argument("--study-dir", required=True, help="Output dir; also holds optuna.db for resume.")
    ap.add_argument("--trials", type=int, default=12, help="trials to run this invocation; 0 re-emits ranking.json.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="build configs + wiring only")
    return ap.parse_args()


def main() -> None:
    run_study(parse_args())


if __name__ == "__main__":
    main()
