"""Build the frozen trial config and its trainer YAML"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import yaml
from training.config import check_keys


# Sampled names that select a template instead of naming a trainer argument, 
SPECIAL_KEYS = {"curriculum_template"}

# Basename of the per-trial YAML, the trainer can be launched with --config trial 
TRIAL_CONFIG_NAME = "trial"


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """`allow_nan=False` rejects NaN/inf rather than writing tokens that are not JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def deep_merge(*configs: Mapping[str, Any]) -> dict[str, Any]:
    """Later configs win. Nested mappings merge key-wise, """
    merged: dict[str, Any] = {}
    for config in configs:
        for key, value in config.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
    return merged



def build_trial_config(
    base_config: Mapping[str, Any],
    stage_config: Mapping[str, Any],
    sampled: Mapping[str, Any],
    run_dir: str | Path,
) -> dict[str, Any]:
    """freeze one trial: base config < stage fixed_args < sampled values < curriculum template"""
    train_args = deep_merge(
        base_config["train_args"],
        stage_config.get("fixed_args", {}),
        {k: v for k, v in sampled.items() if k not in SPECIAL_KEYS},
    )

    if "curriculum_template" in sampled:
        name = sampled["curriculum_template"]
        templates = stage_config["templates"]
        if name not in templates:
            raise ValueError(f"Unknown curriculum_template: {name}")
        train_args.update(templates[name])

    train_args["run_dir"] = str(run_dir)

    return {
        "train_args": train_args,
        "sampled_params": dict(sampled),
        "score": deepcopy(base_config["score"]),
        "stage_description": stage_config.get("description", ""),
    }



def write_trial_yaml(train_args: Mapping[str, Any], trial_dir: str | Path) -> None:
    """Write trial.yaml, refusing any key set the trainer would not accept."""
    check_keys(train_args, "trial train_args")

    trial_dir = Path(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open(trial_dir / f"{TRIAL_CONFIG_NAME}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(train_args), f, sort_keys=True, default_flow_style=False)
        
        
