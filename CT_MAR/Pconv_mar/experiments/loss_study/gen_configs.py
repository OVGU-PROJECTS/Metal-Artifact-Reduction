"""Generate YAML configs for loss sweeps and ablations.
-Configs are written to configs/loss_study_configs 
- results are written under "study_root".
"""
import argparse
import json
import os
from pathlib import Path
import yaml
from training.terms import CALIB_KEY, START_KEY, WEIGHT_KEY

# Study flow: apply calibration -> anchor -> apply gradients -> sweeps -> ablations.
REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(
    os.environ.get("MAR_CONFIG_DIR", REPO / "configs" / "loss_study_configs")
)
BASE = CONFIG_DIR / "pconv.yaml"

# the terms this study sweeps, rmse and hole are held fixed
TERMS = ["img", "edge", "dc", "perc"]
OFF_EPOCH = 1000

# -------------------- Shared config helpers ---------------------------------------

def load_base_config():
    return yaml.safe_load(BASE.read_text())

def save_yaml_config(name, config):
    path = CONFIG_DIR / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"wrote {path}")
    return name

def save_config_manifest(study_root, label, names):
    root = Path(study_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{label}_configs.txt"
    path.write_text("\n".join(names) + ("\n" if names else ""))
    print(f"wrote {path}")

def require_loss_weights(terms):
    base = load_base_config()
    weights = {}
    for term in terms:
        value = float(base.get(WEIGHT_KEY[term], 0.0))
        if value <= 0.0:
            raise SystemExit(
                f"{WEIGHT_KEY[term]} is {value} in pconv.yaml, "
                f"run calib+grad apply before generating {term} configs."
            )
        weights[term] = value
    return weights

def build_variant_config(args, name, init_weights, epochs, active, weights):
    config = load_base_config()
    config.update(run_dir=f"{args.study_root}/{name}", init_weights=init_weights,
                  initial_epoch=0, epochs=epochs, seed=args.seed, bd_every_epochs=1)
    config.update(w_rmse=0.0, w_hole=1.0, start_ep_hole=0)
    for term in TERMS:
        config[WEIGHT_KEY[term]] = 0.0
        config[START_KEY[term]] = OFF_EPOCH
    for term in active:
        config[WEIGHT_KEY[term]] = round(weights[term], 8)
        config[START_KEY[term]] = 0
    return config

def save_loss_variant(args, name, active, weights):
    config = build_variant_config(
        args, name, args.anchor_ckpt, args.branch_epochs, active, weights
    )
    return save_yaml_config(name, config)

# --------------------------------Calibration and anchor -------------------------------------------
def apply_calibration_and_gradient_settings(args):
    config = load_base_config()
    if not args.calib_json and not args.grad_json and not args.dc_mode:
        raise SystemExit("apply requires --calib_json, --grad_json, or --dc_mode")

    if args.dc_mode:
        config["dc_mode"] = args.dc_mode

    if args.calib_json:
        calibration = json.loads(Path(args.calib_json).read_text())["results"]
        calib_terms = (list(CALIB_KEY) if args.calib_terms == "all"
                       else [item.strip() for item in args.calib_terms.split(",")
                             if item.strip()])
        unknown = [term for term in calib_terms if term not in CALIB_KEY]
        if unknown:
            raise SystemExit(
                f"unknown calibration term(s) {unknown}, "
                f"choose from {list(CALIB_KEY)} or all"
            )
        for term in calib_terms:
            attr = CALIB_KEY[term]
            if term in calibration:
                config[attr] = round(float(calibration[term]), 9)

    if args.grad_json:
        gradient = json.loads(Path(args.grad_json).read_text())["results"]
        grad_norm = {name: values["grad_norm"] for name, values in gradient.items()}
        if grad_norm.get("hole", 0) <= 0:
            raise SystemExit("grad_norm(hole) is 0/missing, re-run calibrate grad")
        config["w_rmse"], config["w_hole"] = 0.0, 1.0
        for term in TERMS:
            norm = grad_norm.get(term, 0)
            config[WEIGHT_KEY[term]] = round(grad_norm["hole"] / norm, 8) if norm > 0 else 0.0

    print("applied to pconv.yaml:")
    if args.dc_mode:
        print(f"  dc_mode: {config['dc_mode']}")
    if args.calib_json:
        for term in calib_terms:
            attr = CALIB_KEY[term]
            print(f"  {attr}: {config[attr]}")
    if args.grad_json:
        print(f"  hole_grad: {grad_norm['hole']:.6g}")
        for term in TERMS:
            weighted = config[WEIGHT_KEY[term]] * grad_norm[term]
            print(f"  {WEIGHT_KEY[term]}: {config[WEIGHT_KEY[term]]} (weighted_grad={weighted:.6g})")
    save_yaml_config("pconv", config)

def generate_anchor_config(args):
    config = build_variant_config(
        args, "anchor", "", args.anchor_epochs, [], {}
    )
    save_yaml_config("ls_anchor", config)

# ----------------------- Sweeps and final ablations ----------------------------------------
def generate_sweep_configs(args):
    terms = [item.strip() for item in args.terms.split(",") if item.strip()]
    multipliers = [float(item.strip()) for item in args.mults.split(",")
                   if item.strip()]
    unknown = [term for term in terms if term not in TERMS]
    if unknown:
        raise SystemExit(f"unknown term(s) {unknown}, choose from {TERMS}")
    weights = require_loss_weights(terms)

    names = [] if args.skip_b0 else [save_loss_variant(args, "ls_b0", [], {})]
    for term in terms:
        for multiplier in multipliers:
            tag = f"{float(multiplier):g}".replace("-", "m").replace(".", "p")
            tag = "0" + tag[2:] if tag.startswith("0p") else tag
            name = f"ls_{term}_x{tag}"
            names.append(save_loss_variant(
                args, name, [term], {term: weights[term] * multiplier}
            ))

    save_config_manifest(args.study_root, "pilot", names)

def generate_ablation_configs(args):
    base = load_base_config()
    weights = {term: float(base.get(WEIGHT_KEY[term], 0.0)) for term in TERMS}
    missing = [term for term, value in weights.items() if value <= 0.0]
    if missing:
        raise SystemExit(
            "phase 2 requires positive manually selected weights for every term, "
            f"missing={missing}. Edit {BASE} before running final_ablations."
        )

    names = [save_loss_variant(args, "ls_full", TERMS, weights)]
    for omitted in TERMS:
        active = [term for term in TERMS if term != omitted]
        names.append(save_loss_variant(args, f"ls_loo_{omitted}", active, weights))
    save_config_manifest(args.study_root, "final_ablation", names)
    
###########################################################################################

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--study_root", default="/mnt/public/bahaa/results/loss_study",
                        help="run_dir prefix, each variant get <study_root>/<variant>.")
    common.add_argument("--seed", type=int, default=42)

    anchor = commands.add_parser("anchor", parents=[common])
    anchor.add_argument("--anchor_epochs", type=int, default=20)

    apply = commands.add_parser("apply", parents=[common])
    apply.add_argument("--calib_json", default="", help="output_json from tools.calibrate calib.")
    apply.add_argument("--calib_terms", default="all",
                       help="Comma list of calibration terms to apply, or all (default)")
    apply.add_argument("--grad_json", default="", help="output_json from tools.calibrate grad.")
    apply.add_argument("--dc_mode", default="",
                       help="set pconv.yaml dc_mode before calib/grad/sweeps.")

    final = commands.add_parser("final_ablations", parents=[common])
    sweeps = commands.add_parser("sweeps", parents=[common])
    for command, epochs, checkpoint_help in [(final, 15, "shared phase-1 anchor best.pt."),
                                             (sweeps, 8, "anchor best.pt to warm-start from.")]:
        command.add_argument("--anchor_ckpt", required=True, help=checkpoint_help)
        command.add_argument("--branch_epochs", type=int, default=epochs)

    sweeps.add_argument("--terms", default="img,edge,perc",
                        help="comma list, default excludes dc: img,edge,perc")
    sweeps.add_argument("--mults", default="0.25,0.5,1,2,4",
                        help="comma multipliers of the matched pconv.yaml weights")
    sweeps.add_argument("--skip_b0", action="store_true",
                        help="do not regenerate/retrain ls_b0, useful when branches already trained it")

    args = parser.parse_args()
    if not BASE.exists():
        raise SystemExit(f"base config not found: {BASE} (set MAR_CONFIG_DIR?)")
    handlers = {
        "anchor": generate_anchor_config,
        "apply": apply_calibration_and_gradient_settings,
        "final_ablations": generate_ablation_configs,
        "sweeps": generate_sweep_configs,
    }
    handlers[args.cmd](args)


if __name__ == "__main__":
    main()
