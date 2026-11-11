"""Create trajectory/blur and view-count splits with shared held-out cameras.

Input must be one joint COLMAP/Nerfstudio reconstruction containing all image
conditions. Prefixes are the filename stems used in the merged COLMAP input, e.g.
  orbit_00001.png, spiral_00001.png, blur_moderate_00001.png, holdout_00001.png

The holdout prefix is split once into fixed validation/test sets. Each condition
gets nested train subsets at requested ratios while using those same val/test
images and the same COLMAP coordinate system.
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np


def fps(points: np.ndarray, count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= len(points):
        return list(range(len(points)))
    first = int(np.argmax(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    chosen = [first]
    min_d = np.sum((points - points[first]) ** 2, axis=1)
    while len(chosen) < count:
        j = int(np.argmax(min_d))
        chosen.append(j)
        min_d = np.minimum(min_d, np.sum((points - points[j]) ** 2, axis=1))
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description="Generate condition and view-count splits.")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--condition-prefix", required=True, help="Filename stem prefix, e.g. orbit_")
    p.add_argument("--holdout-prefix", default="holdout_")
    p.add_argument("--val-fraction", type=float, default=0.35, help="Fraction of holdout pool reserved for validation.")
    p.add_argument("--ratios", default="1.0,0.5,0.25,0.10")
    a = p.parse_args()

    base = a.dataset / "transforms.json"
    meta = json.loads(base.read_text(encoding="utf-8"))
    frames = meta.get("frames", [])
    if not frames:
        raise SystemExit("No frames in transforms.json")

    names, poses = [], []
    for f in frames:
        names.append(f["file_path"])
        poses.append(np.asarray(f["transform_matrix"], dtype=float)[:3, 3])
    poses_arr = np.asarray(poses)

    hold = [i for i, n in enumerate(names) if Path(n).name.startswith(a.holdout_prefix)]
    train = [i for i, n in enumerate(names) if Path(n).name.startswith(a.condition_prefix)]

    if len(hold) < 4:
        raise SystemExit(f"Need >=4 registered holdout frames; found {len(hold)}")
    if len(train) < 4:
        print(f"⚠️ Skipping {a.condition_prefix} (only {len(train)} registered frames)")
        return

    test_n = max(2, min(len(hold) - 1, round(len(hold) * (1.0 - a.val_fraction))))
    test_local = fps(poses_arr[hold], test_n)
    test = [hold[j] for j in test_local]

    val_pool = [i for i in hold if i not in set(test)]
    val_local = fps(poses_arr[val_pool], len(val_pool))
    val = [val_pool[j] for j in val_local]

    for r in [float(x) for x in a.ratios.split(",")]:
        n = max(2, round(len(train) * r))
        local = fps(poses_arr[train], min(n, len(train)))
        chosen = [train[j] for j in local]

        out = copy.deepcopy(meta)
        out["train_filenames"] = [names[i] for i in chosen]
        out["val_filenames"] = [names[i] for i in val]
        out["test_filenames"] = [names[i] for i in test]

        out_path = a.dataset / f"transforms_{a.condition_prefix.rstrip('_')}_views_{int(round(r*100)):03d}.json"
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"{out_path.name}: train={len(chosen)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()
