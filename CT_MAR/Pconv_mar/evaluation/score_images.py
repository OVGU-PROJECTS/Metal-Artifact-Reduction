"""Score a prediction folder.
- reads target/, pred/ and mask/ (PConv-MAR, DuDoNet or CNN-MAR)
- writes metrics_image.csv

    python -m evaluation.score_images \
        --pred_dir evaluation/predictions/pconv_fullstack_bestckpt/head --fov head
"""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from metrics.scoring.image import FOV_MM, score_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="out_dir written by run_inference.py")
    ap.add_argument("--fov", required=True, choices=["head", "body"])
    args = ap.parse_args()

    root = Path(args.pred_dir)
    fov_mm = FOV_MM[args.fov]
    targets = sorted((root / "target").glob("nometal_*_512x512.npy"))
    print(f"[info] {len(targets)} cases in {root}, fov={args.fov} ({fov_mm} mm)")

    rows = []
    for target in targets:
        token = target.stem.split("_")[1]
        ref = np.load(target)
        pred = np.load(root / "pred" / target.name)
        metal = np.load(root / "mask" / f"metalonlymask_{token}_512x512.npy") > 0.5
        row = score_case(ref, pred, metal, fov_mm)
        rows.append({"case": token, **row})

    keys = list(rows[0].keys())
    out_csv = root / "metrics_image.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[info] wrote {out_csv}")
    for key in keys[1:]:
        print(f"  {key:26s} {np.mean([r[key] for r in rows]):.5f}")

if __name__ == "__main__":
    main()


