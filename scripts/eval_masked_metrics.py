"""Evaluate Nerfstudio models with strict foreground-mask support and Masked LPIPS.

Standard `ns-eval` evaluates against the RAW camera frames where the background
(floor tiles, walls, room) is present. Because the models were trained with foreground
masks (which ignores background rays and renders solid black/background outside the mask),
evaluating the full frame compares black background against bright room lighting,
artificially collapsing PSNR regardless of model quality.

This script evaluates:
1. `masked_psnr` (Masked-PSNR): Both Prediction and Ground Truth have background zeroed out (black).
2. `fg_psnr` (FG-PSNR): Evaluated strictly inside the object's foreground mask pixels.
3. `masked_ssim`: Structural similarity on foreground cropped bounding box.
4. `masked_lpips`: Learned Perceptual Image Patch Similarity (AlexNet) computed on masked foreground images.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
from pathlib import Path
import numpy as np
import torch
import cv2

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Ensure PyTorch 2.6+ compatibility for nerfstudio checkpoint loading
try:
    import importlib.util
    spec = importlib.util.find_spec("nerfstudio")
    if spec and spec.origin:
        ns_root = Path(spec.origin).parent
        old = 'torch.load(load_path, map_location="cpu")'
        new = 'torch.load(load_path, map_location="cpu", weights_only=False)'
        for rel in ("utils/eval_utils.py", "engine/trainer.py"):
            p = ns_root / rel
            if p.exists():
                t = p.read_text(encoding="utf-8")
                if old in t:
                    p.write_text(t.replace(old, new), encoding="utf-8")
except Exception:
    pass

try:
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.cameras.cameras import Cameras
    NERFSTUDIO_AVAILABLE = True
except ImportError:
    NERFSTUDIO_AVAILABLE = False


def ensure_dataset_exists(config_path: Path) -> None:
    """Tự động kiểm tra và giải nén dữ liệu SFM từ Google Drive nếu ổ đĩa Colab bị thiếu."""
    try:
        content = config_path.read_text(encoding="utf-8", errors="ignore")
        # Tìm đường dẫn data: e.g. /content/project/data/processed/box_001/joint_sfm/transforms_*.json
        import re
        m = re.search(r'data:\s*(?:&id\d+\s*)?(?:!!python/object/apply:pathlib\._local\.PosixPath\s*\n\s*-\s*)?([^\n\r]+)', content)
        if not m:
            return
        data_str = m.group(1).strip().strip("'- \"")
        data_path = Path(data_str)
        if data_path.exists():
            return
        
        # Nếu file transforms không tồn tại, trích xuất object_id từ đường dẫn
        obj_match = re.search(r'processed/([^/]+)/joint_sfm', data_str)
        obj_id = obj_match.group(1) if obj_match else None
        if not obj_id:
            # Fallback lấy từ config path: 06_experiments/<obj_id>/...
            parts = config_path.parts
            if "06_experiments" in parts:
                idx = parts.index("06_experiments")
                if idx + 1 < len(parts):
                    obj_id = parts[idx + 1]

        if not obj_id:
            return

        print(f"⚠️ Dữ liệu SFM tại {data_path} chưa có trên ổ đĩa Colab.")
        # Tìm kiếm file tar lưu trữ trên Google Drive
        drive_roots = [
            Path("/content/drive/MyDrive/CV3D_Final_Project"),
            Path("/content/drive/MyDrive")
        ]
        candidates = []
        for dr in drive_roots:
            candidates.extend([
                dr / "05_processed" / f"processed_{obj_id}_joint_sfm_v001.tar",
                dr / "99_archives" / f"processed_{obj_id}_joint_sfm_v001.tar",
                dr / f"processed_{obj_id}_joint_sfm_v001.tar",
            ])

        tar_file = next((p for p in candidates if p.exists()), None)
        if tar_file is not None:
            dest_dir = data_path.parent.parent # /content/project/data/processed/box_001
            print(f"📦 Đang tự động giải nén bản lưu COLMAP & Mask từ Drive: {tar_file}...")
            dest_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["tar", "-xf", str(tar_file), "-C", str(dest_dir.parent)], check=True)
            print(f"✅ Đã khôi phục thành công dữ liệu vào {dest_dir}!")
        else:
            print(f"❌ Không tìm thấy file lưu trữ tar cho {obj_id} trên Drive!")
    except Exception as e:
        print(f"⚠️ Không thể tự động phục hồi dataset: {e}")


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
    
    if mse <= 0.0:
        return 100.0
    return float(-10.0 * np.log10(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM on 0-255 uint8 images using OpenCV or skimage."""
    try:
        from skimage.metrics import structural_similarity as ssim
        return float(ssim(img1, img2, channel_axis=2, data_range=255))
    except ImportError:
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


def init_lpips_fn(device: torch.device):
    """Khởi tạo metric LPIPS (AlexNet) bằng torchmetrics hoặc thư viện lpips."""
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
        return lambda p, g: lpips_metric(p, g).item()
    except Exception:
        pass
    try:
        from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
        return lambda p, g: float(learned_perceptual_image_patch_similarity(p, g, net_type="alex", normalize=True).item())
    except Exception:
        pass
    try:
        import lpips
        loss_fn = lpips.LPIPS(net="alex").to(device)
        return lambda p, g: float(loss_fn(p * 2.0 - 1.0, g * 2.0 - 1.0).item())
    except Exception:
        pass
    return None


def evaluate_config(config_path: Path, output_json: Path | None = None, save_renders: bool = False) -> dict:
    if not NERFSTUDIO_AVAILABLE:
        raise SystemExit("nerfstudio is not installed in the active Python environment.")

    # Đảm bảo dữ liệu test/mask đã sẵn sàng trên đĩa cục bộ
    ensure_dataset_exists(config_path)

    print(f"Loading pipeline from {config_path}...")
    _, pipeline, _, _ = eval_setup(config_path, eval_num_rays_per_chunk=None, test_mode="test")
    pipeline.model.eval()

    dataset = pipeline.datamanager.eval_dataset
    print(f"Loaded evaluation dataset with {len(dataset)} held-out frames.")

    device = pipeline.device
    lpips_fn = init_lpips_fn(device)
    if lpips_fn is not None:
        print("✅ Đã kích hoạt Masked LPIPS (AlexNet Perceptual Metric).")
    else:
        print("ℹ️ Không tìm thấy torchmetrics/lpips, LPIPS sẽ hiển thị N/A.")

    results_per_frame = []
    fg_psnrs = []
    masked_psnrs = []
    unmasked_psnrs = []
    ssims = []
    lpips_list = []

    render_dir = config_path.parent / "eval_masked_renders"
    if save_renders:
        render_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i in range(len(dataset)):
            # 1. Render novel view từ model đã học
            camera = dataset.cameras[i:i+1].to(device)
            outputs = pipeline.model.get_outputs_for_camera(camera)
            pred_rgb = outputs["rgb"] # (H, W, 3) in [0, 1]
            if pred_rgb.ndim == 4:
                pred_rgb = pred_rgb.squeeze(0)

            # 2. Lấy ảnh Ground Truth
            batch = dataset.get_data(i)
            gt_rgb = batch["image"].to(device)
            if gt_rgb.ndim == 4:
                gt_rgb = gt_rgb.squeeze(0)

            # Khớp kích thước ảnh nếu có chênh lệch tỷ lệ
            if pred_rgb.shape[:2] != gt_rgb.shape[:2]:
                pred_rgb = torch.nn.functional.interpolate(
                    pred_rgb.permute(2, 0, 1).unsqueeze(0),
                    size=gt_rgb.shape[:2],
                    mode="bilinear"
                ).squeeze(0).permute(1, 2, 0)

            # 3. Lấy mask
            mask = None
            if "mask" in batch and batch["mask"] is not None:
                m = batch["mask"]
                if isinstance(m, torch.Tensor):
                    mask = m.to(device)
                else:
                    mask = torch.from_numpy(m).to(device)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(-1)

            # Nếu batch không có mask trực tiếp, nạp từ file ảnh mask
            if mask is None and hasattr(dataset, "image_filenames"):
                img_path = Path(dataset.image_filenames[i])
                possible_masks = [
                    img_path.parent.parent / img_path.parent.name.replace("images", "masks") / f"{img_path.stem}.png",
                    img_path.parent.parent / "masks" / f"{img_path.stem}.png",
                    img_path.parent.parent / "masks_2" / f"{img_path.stem}.png",
                    img_path.parent.parent / "masks_4" / f"{img_path.stem}.png",
                ]
                mask_file = next((p for p in possible_masks if p.is_file()), None)
                if mask_file is not None:
                    m_cv = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    if m_cv is not None:
                        if m_cv.shape[:2] != gt_rgb.shape[:2]:
                            m_cv = cv2.resize(m_cv, (gt_rgb.shape[1], gt_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mask = (torch.from_numpy(m_cv).to(device) >= 128).unsqueeze(-1)

            if mask is None:
                mask = torch.ones_like(gt_rgb[:, :, :1], dtype=torch.bool)
            else:
                mask = (mask > 0.5)

            mask_bool = mask
            mask_3c = mask_bool.expand_as(gt_rgb)

            # 1. Unmasked PSNR (tiêu chuẩn ns-eval)
            unmasked_psnr = calculate_psnr(pred_rgb, gt_rgb, None)
            
            # 2. Foreground-only PSNR (chỉ tính trên các pixel thuộc vật thể)
            fg_psnr = calculate_psnr(pred_rgb, gt_rgb, mask_3c)

            # 3. Masked Full-Frame PSNR (cả GT và Pred đều cắt nền về đen [0,0,0])
            pred_masked = torch.where(mask_3c, pred_rgb, torch.zeros_like(pred_rgb))
            gt_masked = torch.where(mask_3c, gt_rgb, torch.zeros_like(gt_rgb))
            masked_full_psnr = calculate_psnr(pred_masked, gt_masked, None)

            # 4. SSIM trên bounding box của vật thể
            pred_np = (pred_masked.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            gt_np = (gt_masked.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            mask_np = mask_bool.squeeze().cpu().numpy().astype(np.uint8)

            y_indices, x_indices = np.where(mask_np > 0)
            if len(y_indices) > 0:
                y0, y1 = max(0, y_indices.min() - 10), min(mask_np.shape[0], y_indices.max() + 10)
                x0, x1 = max(0, x_indices.min() - 10), min(mask_np.shape[1], x_indices.max() + 10)
                ssim_val = calculate_ssim(pred_np[y0:y1, x0:x1], gt_np[y0:y1, x0:x1])
            else:
                ssim_val = calculate_ssim(pred_np, gt_np)

            # 5. Masked LPIPS (So sánh cảm nhận trực quan trên vùng vật thể đã cắt đen nền)
            lpips_val = None
            if lpips_fn is not None:
                try:
                    p_chw = pred_masked.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
                    g_chw = gt_masked.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
                    lpips_val = lpips_fn(p_chw, g_chw)
                    lpips_list.append(lpips_val)
                except Exception:
                    pass

            fg_psnrs.append(fg_psnr)
            masked_psnrs.append(masked_full_psnr)
            unmasked_psnrs.append(unmasked_psnr)
            ssims.append(ssim_val)

            results_per_frame.append({
                "frame_idx": i,
                "fg_psnr": round(fg_psnr, 2),
                "masked_psnr": round(masked_full_psnr, 2),
                "ssim": round(ssim_val, 4),
                "lpips": round(lpips_val, 4) if lpips_val is not None else None
            })

            if save_renders and i < 5:
                # Lưu ảnh so sánh 2 cột: GT Masked | Pred Masked
                side_by_side = np.hstack([
                    cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
                ])
                cv2.imwrite(str(render_dir / f"compare_frame_{i:04d}.jpg"), side_by_side)

    mean_lpips = round(float(np.mean(lpips_list)), 4) if lpips_list else None
    summary = {
        "metrics": {
            "masked_psnr_mean": round(float(np.mean(masked_psnrs)), 2),
            "fg_psnr_mean": round(float(np.mean(fg_psnrs)), 2),
            "masked_ssim_mean": round(float(np.mean(ssims)), 4),
            "masked_lpips_mean": mean_lpips,
            # Tương thích ngược với các bản đọc cũ
            "foreground_only_psnr_mean": round(float(np.mean(fg_psnrs)), 2),
            "masked_full_psnr_mean": round(float(np.mean(masked_psnrs)), 2),
            "unmasked_raw_psnr_mean": round(float(np.mean(unmasked_psnrs)), 2),
            "ssim_mean": round(float(np.mean(ssims)), 4),
        },
        "num_frames_evaluated": len(dataset)
    }

    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ ĐÁNH GIÁ MASKED METRICS (FOREGROUND-ONLY)")
    print("=" * 60)
    print(f"🎯 Masked-PSNR (Cắt đen nền cả 2):     {summary['metrics']['masked_psnr_mean']} dB")
    print(f"🔍 FG-PSNR (Chỉ tính pixel vật thể):   {summary['metrics']['fg_psnr_mean']} dB")
    print(f"✨ Masked SSIM:                         {summary['metrics']['masked_ssim_mean']}")
    lpips_display = f"{mean_lpips:.4f}" if mean_lpips is not None else "N/A"
    print(f"👁️ Masked LPIPS (AlexNet Perceptual):   {lpips_display}")
    print("=" * 60)

    out_path = output_json or (config_path.parent / "test_metrics_masked.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote detailed metrics to {out_path}")

    compat_path = config_path.parent / "masked_eval_metrics.json"
    if compat_path != out_path:
        compat_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Nerfstudio checkpoint with foreground masking and LPIPS.")
    parser.add_argument("--config", "--config-path", dest="config", type=Path, default=None, help="Path to config.yml")
    parser.add_argument("--output-json", type=Path, default=None, help="Where to save metrics JSON")
    parser.add_argument("--save-renders", action="store_true", help="Save side-by-side comparison renders")
    parser.add_argument("--object-id", type=str, default=None, help="Evaluate all experiments for an object (e.g. box_001)")
    parser.add_argument("--all-objects", action="store_true", help="Evaluate all objects in 06_experiments")
    parser.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/CV3D_Final_Project"), help="Google Drive base path")
    args = parser.parse_args()

    if args.config is not None:
        evaluate_config(args.config, args.output_json, args.save_renders)
    elif args.object_id or args.all_objects:
        objs = ['box_001', 'basket_001', 'figure_001', 'stuff_toy_001'] if args.all_objects else [args.object_id]
        for obj in objs:
            obj_exp = args.drive_dir / '06_experiments' / obj
            if not obj_exp.exists():
                continue
            cfgs = sorted(obj_exp.glob('**/config.yml'))
            print(f"Found {len(cfgs)} configs for {obj} in {obj_exp}")
            for cfg in cfgs:
                eval_out = cfg.parent / 'test_metrics_masked.json'
                evaluate_config(cfg, eval_out, args.save_renders)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
