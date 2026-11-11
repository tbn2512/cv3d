"""Evaluate Nerfstudio models (nerfacto & splatfacto) with foreground masks, LPIPS, and FPS."""
from __future__ import annotations
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import cv2

# Ensure torch.load uses weights_only=False on PyTorch 2.6+ / 2.11
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

from nerfstudio.utils.eval_utils import eval_setup


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        valid = mask.bool()
        if valid.sum() == 0:
            return 0.0
        diff = (img1[valid].float() - img2[valid].float()) ** 2
        mse = torch.mean(diff).item()
    else:
        mse = torch.mean((img1.float() - img2.float()) ** 2).item()
    if mse <= 1e-10:
        return 100.0
    return float(-10.0 * np.log10(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim
    return float(ssim(img1, img2, channel_axis=2, data_range=255))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Nerfstudio checkpoint with foreground masking, LPIPS, and FPS.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yml")
    parser.add_argument("--output-json", type=Path, default=None, help="Where to save metrics JSON")
    parser.add_argument("--save-renders", action="store_true", help="Save side-by-side comparison renders")
    args = parser.parse_args()

    print(f"Loading pipeline from {args.config}...")
    _, pipeline, _, _ = eval_setup(args.config, eval_num_rays_per_chunk=None, test_mode="test")
    pipeline.model.eval()
    device = pipeline.device

    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device)
        lpips_fn.eval()
    except Exception:
        pass

    dataset = pipeline.datamanager.eval_dataset
    num_frames = len(dataset)
    print(f"Loaded evaluation dataset with {num_frames} held-out frames.")

    fg_psnrs, masked_psnrs, unmasked_psnrs, ssims, lpips_vals, render_times = [], [], [], [], [], []
    render_dir = args.config.parent / "eval_masked_renders"
    if args.save_renders:
        render_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i in range(num_frames):
            camera = dataset.cameras[i:i+1].to(device)
            # Call get_data(i) with default image_type="float32"
            batch = dataset.get_data(i)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = pipeline.model.get_outputs_for_camera(camera)
            torch.cuda.synchronize()
            render_times.append(time.perf_counter() - t0)

            pred_rgb = outputs["rgb"].to(device).float()
            gt_rgb = batch["image"].to(device)
            if gt_rgb.dtype == torch.uint8 or gt_rgb.max() > 1.5:
                gt_rgb = gt_rgb.float() / 255.0
            else:
                gt_rgb = gt_rgb.float()

            if pred_rgb.shape[:2] != gt_rgb.shape[:2]:
                pred_rgb = torch.nn.functional.interpolate(
                    pred_rgb.permute(2, 0, 1).unsqueeze(0),
                    size=gt_rgb.shape[:2],
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)

            if "mask" in batch and batch["mask"] is not None:
                mask = batch["mask"].to(device)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(-1)
                if mask.shape[:2] != gt_rgb.shape[:2]:
                    mask = torch.nn.functional.interpolate(
                        mask.float().permute(2, 0, 1).unsqueeze(0),
                        size=gt_rgb.shape[:2],
                        mode="nearest"
                    ).squeeze(0).permute(1, 2, 0)
            else:
                mask = torch.ones_like(gt_rgb[:, :, :1], dtype=torch.bool)

            mask_bool = (mask > 0.5)
            mask_3c = mask_bool.expand_as(gt_rgb)

            unmasked_psnr = calculate_psnr(pred_rgb, gt_rgb, None)
            fg_psnr = calculate_psnr(pred_rgb, gt_rgb, mask_3c)

            pred_masked = torch.where(mask_3c, pred_rgb, torch.zeros_like(pred_rgb))
            gt_masked = torch.where(mask_3c, gt_rgb, torch.zeros_like(gt_rgb))
            masked_full_psnr = calculate_psnr(pred_masked, gt_masked, None)

            pred_np = (pred_masked.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            gt_np = (gt_masked.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            mask_np = mask_bool.squeeze().cpu().numpy().astype(np.uint8)

            y_indices, x_indices = np.where(mask_np > 0)
            if len(y_indices) > 0:
                y0, y1 = max(0, y_indices.min() - 10), min(mask_np.shape[0], y_indices.max() + 10)
                x0, x1 = max(0, x_indices.min() - 10), min(mask_np.shape[1], x_indices.max() + 10)
                ssim_val = calculate_ssim(pred_np[y0:y1, x0:x1], gt_np[y0:y1, x0:x1])
            else:
                ssim_val = calculate_ssim(pred_np, gt_np)

            if lpips_fn is not None:
                p_t = (pred_masked.clamp(0, 1).permute(2, 0, 1).unsqueeze(0) * 2.0) - 1.0
                g_t = (gt_masked.clamp(0, 1).permute(2, 0, 1).unsqueeze(0) * 2.0) - 1.0
                lpips_vals.append(float(lpips_fn(p_t, g_t).item()))

            fg_psnrs.append(fg_psnr)
            masked_psnrs.append(masked_full_psnr)
            unmasked_psnrs.append(unmasked_psnr)
            ssims.append(ssim_val)

            if args.save_renders and i < 5:
                side_by_side = np.hstack([
                    cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
                ])
                cv2.imwrite(str(render_dir / f"compare_frame_{i:04d}.jpg"), side_by_side)

    mean_render_sec = float(np.mean(render_times)) if render_times else 1.0
    fps = round(1.0 / max(1e-6, mean_render_sec), 2)

    summary = {
        "metrics": {
            "foreground_only_psnr_mean": round(float(np.mean(fg_psnrs)), 2),
            "masked_full_psnr_mean": round(float(np.mean(masked_psnrs)), 2),
            "unmasked_raw_psnr_mean": round(float(np.mean(unmasked_psnrs)), 2),
            "ssim_mean": round(float(np.mean(ssims)), 4),
            "lpips_mean": round(float(np.mean(lpips_vals)), 4) if lpips_vals else None,
            "render_fps": fps
        },
        "num_frames_evaluated": num_frames
    }

    print(f"🎯 FG PSNR: {summary['metrics']['foreground_only_psnr_mean']} dB | Masked PSNR: {summary['metrics']['masked_full_psnr_mean']} dB | SSIM: {summary['metrics']['ssim_mean']} | LPIPS: {summary['metrics']['lpips_mean']} | FPS: {fps}")
    out_path = args.output_json or (args.config.parent / "masked_eval_metrics.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
