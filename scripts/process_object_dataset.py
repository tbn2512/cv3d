"""
End-to-end preprocessing, curation, masking, and archiving script for CV3D objects.
Handles frame extraction, auditing, auto-curation, merging, parallel IS-Net masking,
tar archiving with SHA-256 updates, metadata YAML generation, and temp disk cleanup.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

python_exe = sys.executable

# Conditions mapping: (folder_suffix, condition_key, prefix, target_frames)
CONDITIONS = [
    ("orbit_clean", "orbit_clean", "orbit", 80),
    ("spiral_clean", "spiral_clean", "spiral", 80),
    ("blur_moderate", "blur_moderate", "blur_moderate", 80),
    ("blur_heavy", "blur_heavy", "blur_heavy", 80),
    ("holdout", "holdout", "holdout", 42),
]

def find_video(raw_dir: Path, suffix: str) -> Path:
    cond_dir = raw_dir / f"{raw_dir.name}_{suffix}"
    if not cond_dir.is_dir():
        cond_dir = raw_dir / suffix
    if not cond_dir.is_dir():
        raise FileNotFoundError(f"Could not find video directory for condition {suffix} in {raw_dir}")
    movs = list(cond_dir.glob("*.mov")) + list(cond_dir.glob("*.mp4"))
    if not movs:
        raise FileNotFoundError(f"No .mov/.mp4 found in {cond_dir}")
    return movs[0]

def process_condition_stream(obj_id: str, raw_dir: Path, cond_tuple, candidates_base: Path, curated_base: Path, manifests_dir: Path, colmap_images_dir: Path, fps: float = 2.0):
    suffix, cond_key, prefix, target = cond_tuple
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH.")
        
    vid = find_video(raw_dir, suffix)
    cand_cond = candidates_base / suffix
    curated_cond = curated_base / suffix
    csv_path = manifests_dir / f"{obj_id}_{suffix}_audit.csv"
    
    cand_cond.mkdir(parents=True, exist_ok=True)
    curated_cond.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract candidate frames
    pattern = cand_cond / "frame_%05d.png"
    cmd = [
        ffmpeg, "-hide_banner", "-y", "-i", str(vid),
        "-vf", f"fps={fps}", "-fps_mode", "passthrough",
        "-pix_fmt", "rgb24", "-compression_level", "3",
        str(pattern)
    ]
    print(f"\n[{obj_id}] Extracting {vid.name} -> {cand_cond}...", flush=True)
    subprocess.run(cmd, check=True)
    extracted = len(list(cand_cond.glob("frame_*.png")))
    print(f"  Extracted {extracted} candidate frames for {suffix}.", flush=True)
    
    # 2. Audit
    cmd_audit = [python_exe, "scripts/audit_images.py", "--input", str(cand_cond), "--csv", str(csv_path)]
    subprocess.run(cmd_audit, check=True)
    
    # 3. Auto-curate
    cmd_curate = [python_exe, "scripts/auto_curate.py", "--audit-csv", str(csv_path), "--condition", cond_key, "--target", str(target)]
    subprocess.run(cmd_curate, check=True)
    
    # 4. Select frames
    cmd_select = [python_exe, "scripts/select_frames.py", "--audit-csv", str(csv_path), "--output", str(curated_cond)]
    subprocess.run(cmd_select, check=True)
    
    # 5. Merge into colmap input with standard prefix
    curated_files = sorted(list(curated_cond.glob("*.png")))
    for i, f in enumerate(curated_files, 1):
        dst = colmap_images_dir / f"{prefix}_{i:05d}.png"
        shutil.copy2(f, dst)
    print(f"  Merged {len(curated_files)}/{target} frames with prefix '{prefix}_' into colmap_input.", flush=True)
    
    # 6. Immediate cleanup of condition's temp candidate & curated files
    shutil.rmtree(cand_cond, ignore_errors=True)
    shutil.rmtree(curated_cond, ignore_errors=True)
    _, _, free = shutil.disk_usage(Path(".").resolve())
    print(f"  Reclaimed temp files. Current Free Disk Space: {free / (1024**3):.2f} GiB", flush=True)

def init_rembg_worker():
    global session
    from rembg import new_session
    os.environ['U2NET_HOME'] = str(Path('checkpoints').resolve())
    session = new_session('isnet-general-use')

def process_single_mask(item):
    idx, img_path, out_mask_path = item
    from rembg import remove
    with Image.open(img_path) as im:
        rgba = remove(im.convert('RGB'), session=session).convert('RGBA')
        alpha = np.asarray(rgba.getchannel('A'))
    # Strict binary single-channel mask (0 or 255)
    mask = np.where(alpha >= 127, 255, 0).astype(np.uint8)
    
    # Save mask
    Image.fromarray(mask, mode='L').save(out_mask_path)
    return img_path.name

def step4_generate_masks(obj_id: str, colmap_images_dir: Path, masks_dir: Path):
    print(f"\n=======================================================", flush=True)
    print(f"[{obj_id}] STEP 4: PARALLEL IS-NET MASK GENERATION", flush=True)
    print(f"=======================================================", flush=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(list(colmap_images_dir.glob("*.png")))
    total = len(images)
    print(f"Generating masks for {total} images into {masks_dir}...", flush=True)
    
    tasks = [(idx, p, masks_dir / p.name) for idx, p in enumerate(images)]
    workers = min(6, os.cpu_count() or 4)
    
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=init_rembg_worker) as executor:
        for fname in executor.map(process_single_mask, tasks):
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  [{done}/{total}] Segmented {fname}", flush=True)
    print(f"All {total} masks successfully generated in {masks_dir}!", flush=True)

def step5_package_archive(obj_id: str, colmap_images_dir: Path, masks_dir: Path, archives_dir: Path):
    print(f"\n=======================================================", flush=True)
    print(f"[{obj_id}] STEP 5: PACKAGING DATASET TAR ARCHIVE", flush=True)
    print(f"=======================================================", flush=True)
    archives_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archives_dir / f"dataset_{obj_id}.tar"
    
    images = sorted(list(colmap_images_dir.glob("*.png")))
    masks = sorted(list(masks_dir.glob("*.png")))
    
    print(f"Writing {len(images)} images + {len(masks)} masks to {archive_path}...", flush=True)
    with tarfile.open(archive_path, "w") as tar:
        for p in images:
            tar.add(p, arcname=f"{obj_id}/images/{p.name}")
        for p in masks:
            tar.add(p, arcname=f"{obj_id}/masks/{p.name}")
            
    size_bytes = archive_path.stat().st_size
    h = hashlib.sha256()
    with archive_path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    sha = h.hexdigest()
    print(f"Archive created: {archive_path.name} | Size: {size_bytes / (1024**3):.2f} GiB | SHA-256: {sha}", flush=True)
    
    # Update archives_sha256.csv
    csv_path = archives_dir / "archives_sha256.csv"
    rows = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("archive_name") != archive_path.name]
    rows.append({"archive_name": archive_path.name, "size_bytes": str(size_bytes), "sha256": sha})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["archive_name", "size_bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    print(f"Updated {csv_path}", flush=True)

def step6_cleanup_intermediates(candidates_dir: Path, curated_dir: Path):
    print(f"\n=======================================================", flush=True)
    print(f"STEP 6: RECLAIMING DISK SPACE", flush=True)
    print(f"=======================================================", flush=True)
    for folder in [candidates_dir, curated_dir]:
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
            print(f"  Removed intermediate directory: {folder}", flush=True)
    total, used, free = shutil.disk_usage(Path(".").resolve())
    print(f"Current Free Disk Space: {free / (1024**3):.2f} GiB", flush=True)

def main():
    parser = argparse.ArgumentParser(description="End-to-end dataset preparation for an object.")
    parser.add_argument("--object-id", required=True, choices=["box_1", "shoe_1", "book_stack_1", "book_stack_2"])
    args = parser.parse_args()
    
    obj_id = args.object_id
    raw_dir = Path(f"data/raw/{obj_id}")
    candidates_dir = Path(f"data/candidates/{obj_id}")
    curated_dir = Path(f"data/curated/{obj_id}")
    manifests_dir = Path(f"data/manifests")
    colmap_images_dir = Path(f"data/colmap_input/{obj_id}/images")
    masks_dir = Path(f"data/masks/{obj_id}")
    archives_dir = Path("99_archives")
    
    print(f"Starting end-to-end pipeline for object: {obj_id}", flush=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    colmap_images_dir.mkdir(parents=True, exist_ok=True)
    # Clear any previous images
    for f in colmap_images_dir.glob("*.png"):
        f.unlink()
        
    print(f"\n=======================================================", flush=True)
    print(f"[{obj_id}] STEPS 1-3: STREAMING CANDIDATES, CURATING, & MERGING", flush=True)
    print(f"=======================================================", flush=True)
    for cond in CONDITIONS:
        process_condition_stream(obj_id, raw_dir, cond, candidates_dir, curated_dir, manifests_dir, colmap_images_dir, fps=2.0)
        
    merged_count = len(list(colmap_images_dir.glob("*.png")))
    print(f"\nTotal curated frames merged for {obj_id}: {merged_count} (expected 362)", flush=True)
    if merged_count != 362:
        print(f"WARNING: Expected 362 images, found {merged_count}", flush=True)
        
    step4_generate_masks(obj_id, colmap_images_dir, masks_dir)
    step5_package_archive(obj_id, colmap_images_dir, masks_dir, archives_dir)
    step6_cleanup_intermediates(candidates_dir, curated_dir)
    print(f"\n>>> ALL STEPS FOR {obj_id} COMPLETED SUCCESSFULLY! <<<\n", flush=True)

if __name__ == "__main__":
    main()
