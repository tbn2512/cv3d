"""Evaluate Nerfstudio models with strict foreground-mask support.

Standard `ns-eval` evaluates against the RAW camera frames where the background
(floor tiles, walls, room) is present. Because the models were trained with foreground
masks (which ignores background rays and renders solid black/background outside the mask),
evaluating the full frame compares black background against bright room lighting,
artificially collapsing PSNR regardless of model quality.

This script evaluates:
1. `fg_psnr`: True reconstruction PSNR strictly inside the object's masked pixels.
2. `masked_full_psnr`: PSNR where both Ground Truth and Prediction have background masked to black.
3. `unmasked_raw_psnr`: Baseline unmasked metric (matches default ns-eval behavior for diagnosis).
4. `fg_ssim`: SSIM computed over the bounding box of the masked object.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import cv2

try:
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.cameras.cameras import Cameras
    NERFSTUDIO_AVAILABLE = True
except ImportError:
    NERFSTUDIO_AVAILABLE = False


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Compute PSNR. If mask is provided (boolean or 0/1), compute only over mask."""
    if mask is not None:
        valid = mask.bool()
        if valid.sum() == 0:
            return 0.0
        diff = (img1[valid] - img2[valid]) ** 2
        mse = torch.mean(diff).item()
    else:
        mse = torch.mean((img1 - img2) ** 2).item()
    
    if mse == 0.0:
        return 100.0
    return float(-10.0 * np.log10(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM on 0-255 uint8 images using OpenCV or skimage."""
    try:
        from skimage.metrics import structural_similarity as ssim
        return float(ssim(img1, img2, channel_axis=2, data_range=255))
    except ImportError:
        # Fallback luminance SSIM
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        y1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float64)
        y2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float64)
        mu1 = cv2.GaussianBlur(y1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(y2, (11, 11), 1.5)
        sigma1_sq = cv2.GaussianBlur(y1 ** 2, (11, 11), 1.5) - mu1 ** 2
        sigma2_sq = cv2.GaussianBlur(y2 ** 2, (11, 11), 1.5) - mu2 ** 2
        sigma12 = cv2.GaussianBlur(y1 * y2, (11, 11), 1.5) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(np.mean(ssim_map))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Nerfstudio checkpoint with foreground masking.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yml")
    parser.add_argument("--output-json", type=Path, default=None, help="Where to save metrics JSON")
    parser.add_argument("--save-renders", action="store_true", help="Save side-by-side comparison renders")
    args = parser.parse_args()

    if not NERFSTUDIO_AVAILABLE:
        raise SystemExit("nerfstudio is not installed in the active Python environment.")

    print(f"Loading pipeline from {args.config}...")
    _, pipeline, _, _ = eval_setup(args.config, eval_num_rays_per_chunk=None, test_mode="test")
    pipeline.model.eval()

    dataset = pipeline.datamanager.eval_dataset
    print(f"Loaded evaluation dataset with {len(dataset)} held-out frames.")

    results_per_frame = []
    fg_psnrs = []
    masked_psnrs = []
    unmasked_psnrs = []
    ssims = []

    render_dir = args.config.parent / "eval_masked_renders"
    if args.save_renders:
        render_dir.mkdir(parents=True, exist_ok=True)

    device = pipeline.device

    with torch.no_grad():
        for i in range(len(dataset)):
            camera_ray_bundle, batch = dataset.get_camera_ray_bundle_and_batch(i)
            # Render novel view
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            pred_rgb = outputs["rgb"] # (H, W, 3) in [0, 1]
            gt_rgb = batch["image"].to(device) # (H, W, 3) in [0, 1]

            # Get mask
            if "mask" in batch and batch["mask"] is not None:
                mask = batch["mask"].to(device) # (H, W, 1) or (H, W)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(-1)
            else:
                mask = torch.ones_like(gt_rgb[:, :, :1], dtype=torch.bool)

            # Ensure mask is boolean
            mask_bool = (mask > 0.5)

            # 1. Unmasked PSNR (standard ns-eval)
            unmasked_psnr = calculate_psnr(pred_rgb, gt_rgb, None)
            
            # 2. Foreground-only PSNR (strictly pixels where mask is true)
            mask_3c = mask_bool.expand_as(gt_rgb)
            fg_psnr = calculate_psnr(pred_rgb, gt_rgb, mask_3c)

            # 3. Masked Full-Frame PSNR (both GT and Pred zeroed outside mask)
            pred_masked = torch.where(mask_3c, pred_rgb, torch.zeros_like(pred_rgb))
            gt_masked = torch.where(mask_3c, gt_rgb, torch.zeros_like(gt_rgb))
            masked_full_psnr = calculate_psnr(pred_masked, gt_masked, None)

            # 4. SSIM on cropped bounding box of foreground
            pred_np = (pred_masked.cpu().numpy() * 255).astype(np.uint8)
            gt_np = (gt_masked.cpu().numpy() * 255).astype(np.uint8)
            mask_np = mask_bool.squeeze().cpu().numpy().astype(np.uint8)

            y_indices, x_indices = np.where(mask_np > 0)
            if len(y_indices) > 0:
                y0, y1 = max(0, y_indices.min() - 10), min(mask_np.shape[0], y_indices.max() + 10)
                x0, x1 = max(0, x_indices.min() - 10), min(mask_np.shape[1], x_indices.max() + 10)
                ssim_val = calculate_ssim(pred_np[y0:y1, x0:x1], gt_np[y0:y1, x0:x1])
            else:
                ssim_val = calculate_ssim(pred_np, gt_np)

            fg_psnrs.append(fg_psnr)
            masked_psnrs.append(masked_full_psnr)
            unmasked_psnrs.append(unmasked_psnr)
            ssims.append(ssim_val)

            results_per_frame.append({
                "frame_idx": i,
                "fg_psnr": round(fg_psnr, 2),
                "masked_full_psnr": round(masked_full_psnr, 2),
                "unmasked_raw_psnr": round(unmasked_psnr, 2),
                "ssim": round(ssim_val, 4)
            })

            if args.save_renders and i < 5:
                # Save side-by-side comparison for the first 5 frames
                side_by_side = np.hstack([
                    cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
                ])
                cv2.imwrite(str(render_dir / f"compare_frame_{i:04d}.jpg"), side_by_side)

    summary = {
        "metrics": {
            "foreground_only_psnr_mean": round(float(np.mean(fg_psnrs)), 2),
            "masked_full_psnr_mean": round(float(np.mean(masked_psnrs)), 2),
            "unmasked_raw_psnr_mean": round(float(np.mean(unmasked_psnrs)), 2),
            "ssim_mean": round(float(np.mean(ssims)), 4)
        },
        "num_frames_evaluated": len(dataset),
        "diagnosis_note": "If unmasked_raw_psnr is ~11-12 dB but foreground_only_psnr is >22 dB, the model reconstructed the object properly; the low unmasked score was purely due to ns-eval penalizing the masked-out room background."
    }

    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"🎯 Foreground-Only PSNR (Target Object):  {summary['metrics']['foreground_only_psnr_mean']} dB")
    print(f"🎭 Masked-Frame PSNR (Zeroed Background): {summary['metrics']['masked_full_psnr_mean']} dB")
    print(f"❌ Unmasked Raw PSNR (Standard ns-eval):   {summary['metrics']['unmasked_raw_psnr_mean']} dB")
    print(f"✨ Masked SSIM:                           {summary['metrics']['ssim_mean']}")
    print("=" * 60)

    out_path = args.output_json or (args.config.parent / "masked_eval_metrics.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote detailed metrics to {out_path}")


if __name__ == "__main__":
    main()
