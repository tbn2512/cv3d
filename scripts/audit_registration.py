"""Report COLMAP/Nerfstudio image registration by capture-condition prefix.

Example:
  python scripts/audit_registration.py \
      --input-images data/colmap_input/images \
      --transforms data/processed/joint_sfm/transforms.json \
      --output-csv data/manifests/registration_by_condition.csv

The input directory is the complete submitted image set. transforms.json contains
only images registered by COLMAP, so this computes registered/submitted counts.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_PREFIXES = ("orbit_", "spiral_", "blur_1_50_", "blur_1_25_", "holdout_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-images", type=Path, required=True)
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES))
    args = parser.parse_args()

    if not args.input_images.is_dir():
        raise SystemExit(f"Input image directory not found: {args.input_images}")
    if not args.transforms.is_file():
        raise SystemExit(f"Transforms JSON not found: {args.transforms}")

    import json
    meta = json.loads(args.transforms.read_text(encoding="utf-8"))
    submitted = sorted(p.name for p in args.input_images.iterdir()
                       if p.is_file() and p.suffix.lower() in EXTS)
    registered = {Path(frame["file_path"]).name for frame in meta.get("frames", [])}
    prefixes = [x.strip() for x in args.prefixes.split(",") if x.strip()]
    if not submitted:
        raise SystemExit("No supported image files found in input directory")

    rows = []
    total_registered = 0
    for prefix in prefixes:
        group_submitted = [n for n in submitted if n.startswith(prefix)]
        group_registered = [n for n in group_submitted if n in registered]
        n_sub, n_reg = len(group_submitted), len(group_registered)
        total_registered += n_reg
        fraction = n_reg / n_sub if n_sub else 0.0
        rows.append({"condition_prefix": prefix, "submitted": n_sub,
                     "registered": n_reg, "registration_fraction": f"{fraction:.6f}"})

    ungrouped = [n for n in submitted if not any(n.startswith(p) for p in prefixes)]
    if ungrouped:
        rows.append({"condition_prefix": "UNMATCHED", "submitted": len(ungrouped),
                     "registered": sum(n in registered for n in ungrouped),
                     "registration_fraction": f"{sum(n in registered for n in ungrouped)/len(ungrouped):.6f}"})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition_prefix", "submitted", "registered", "registration_fraction"])
        writer.writeheader()
        writer.writerows(rows)

    total_submitted = len(submitted)
    total_registered = len(registered.intersection(submitted))
    overall = total_registered / total_submitted
    print(f"Overall registration: {total_registered}/{total_submitted} = {overall:.1%}")
    for row in rows:
        print(f"{row['condition_prefix']}: {row['registered']}/{row['submitted']} = {float(row['registration_fraction']):.1%}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
