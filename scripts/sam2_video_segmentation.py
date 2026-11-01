"""
Production SAM 2 Video Prediction Pipeline.
Applies Meta's SAM 2 Video Predictor with temporal memory across all video trajectory sequences
to segment objects cleanly while excluding pedestals/stools.
"""
import os
import shutil
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_fill_holes
from sam2.build_sam import build_sam2_video_predictor

# General background and stool negative anchor points (fixed studio setup)
COMMON_NEG_POINTS = [
    [1080, 500],   # ceiling / kitchen top
    [1080, 750],   # upper background
    [1080, 2600],  # stool top rim center
    [1080, 2750],  # stool tray / disk
    [550, 2800],   # left stool leg upper
    [550, 3100],   # left stool leg mid
    [550, 3400],   # left stool leg foot
    [1600, 2800],  # right stool leg upper
    [1600, 3100],  # right stool leg mid
    [1600, 3400],  # right stool leg foot
    [1080, 3500],  # wooden table floor
    [200, 2000],   # far left background
    [1950, 2000],  # far right background
]

# Positive anchor points per object (centered on the object above stool)
OBJECT_CONFIGS = {
    'box_001': {
        'pos_points': [
            [1080, 1600],
            [1080, 1900],
            [1080, 2200],
            [850, 1900],
            [1300, 1900],
        ]
    },
    'basket_001': {
        'pos_points': [
            [1080, 1500], # handle top
            [1080, 1900], # center mesh
            [800, 2000],  # left mesh
            [1350, 2000], # right mesh
            [1450, 1300], # ribbon flower
        ]
    },
    'stuff_toy_001': {
        'pos_points': [
            [1080, 1900], # center white body
            [1080, 1500], # head
            [1150, 1350], # red mane
            [1150, 1750], # red scarf
            [900, 2000],  # front body
            [1080, 2150], # lower paws
        ]
    }
}

CONDITIONS = ['orbit', 'spiral', 'blur_1_50', 'blur_1_25', 'holdout']

def process_sequence(predictor, obj_id, cond, images_dir, out_mask_dir, preview_dir):
    pattern = f"{cond}_*.png"
    frames = sorted(list(images_dir.glob(pattern)))
    if not frames:
        print(f"[{obj_id}] No frames found for {cond}")
        return
        
    print(f"\n==========================================")
    print(f"Processing {obj_id} | Condition: {cond} ({len(frames)} frames)")
    print(f"==========================================")
    
    # 1. Prepare temporary JPEG frames directory
    temp_dir = Path(f"scratch/temp_frames_{obj_id}_{cond}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, f in enumerate(frames):
        im = Image.open(f)
        im.save(temp_dir / f"{idx:05d}.jpg", quality=95)
        
    # 2. Init video inference state (CPU offloading keeps VRAM < 2GB)
    inference_state = predictor.init_state(
        video_path=str(temp_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True
    )
    
    # 3. Points prompt on frame 0
    cfg = OBJECT_CONFIGS[obj_id]
    pos = cfg['pos_points']
    neg = COMMON_NEG_POINTS
    pts = np.array(pos + neg, dtype=np.float32)
    lbls = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)
    
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=pts,
        labels=lbls
    )
    
    # 4. Propagate across the video sequence
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    preview_cond_dir = preview_dir / cond
    preview_cond_dir.mkdir(parents=True, exist_ok=True)
    
    first_img = cv2.imread(str(frames[0]))
    h_orig, w_orig = first_img.shape[:2]
    scale = 0.25
    
    print(f"Propagating with temporal memory across {len(frames)} frames...")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        raw_m = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(bool)
        # Fill internal holes inside the segmented object
        solid_m = binary_fill_holes(raw_m).astype(np.uint8) * 255
        
        if (solid_m.shape[1], solid_m.shape[0]) != (w_orig, h_orig):
            solid_m = cv2.resize(solid_m, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
        orig_file = frames[out_frame_idx]
        out_mask_file = out_mask_dir / orig_file.name
        cv2.imwrite(str(out_mask_file), solid_m)
        
        # Save keyframe previews for QC (first, quarter, half, three-quarters, last)
        save_checkpoints = {0, len(frames)//4, len(frames)//2, (len(frames)*3)//4, len(frames)-1}
        if out_frame_idx in save_checkpoints:
            img = cv2.imread(str(orig_file))
            small_i = cv2.resize(img, (int(w_orig*scale), int(h_orig*scale)))
            small_m = cv2.resize(solid_m, (int(w_orig*scale), int(h_orig*scale)))
            ov = small_i.copy()
            ov[small_m > 0] = [0, 255, 0] # Green overlay
            bl = cv2.addWeighted(small_i, 0.6, ov, 0.4, 0)
            cv2.imwrite(str(preview_cond_dir / f"{orig_file.stem}_blend.jpg"), bl)
            
    print(f"Completed {cond}: {len(frames)} masks saved to {out_mask_dir}")
    # Cleanup temp frames
    shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    print("Initializing SAM 2.1 Video Predictor on CUDA...")
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    ckpt_path = "checkpoints/sam2.1_hiera_base_plus.pt"
    predictor = build_sam2_video_predictor(model_cfg, ckpt_path, device="cuda")
    
    # 1. Figure_001 is already good -> copy draft to final
    fig_draft = Path("data/masks/figure_001/draft")
    fig_final = Path("data/masks/figure_001/final")
    fig_final.mkdir(parents=True, exist_ok=True)
    for m in fig_draft.glob("*.png"):
        shutil.copy2(m, fig_final / m.name)
    print(f"figure_001: Preserved {len(list(fig_final.glob('*.png')))} masks in final/")
    
    # 2. Process the other three objects with SAM 2 Video Predictor
    for obj_id in ['box_001', 'basket_001', 'stuff_toy_001']:
        images_dir = Path(f"data/colmap_input/{obj_id}/images")
        final_dir = Path(f"data/masks/{obj_id}/final")
        preview_dir = Path(f"scratch/qc_previews/{obj_id}")
        
        for cond in CONDITIONS:
            process_sequence(
                predictor=predictor,
                obj_id=obj_id,
                cond=cond,
                images_dir=images_dir,
                out_mask_dir=final_dir,
                preview_dir=preview_dir
            )
            
    print("\n==========================================")
    print("ALL SAM 2 VIDEO SEGMENTATIONS COMPLETE!")
    print("==========================================")

if __name__ == "__main__":
    main()
