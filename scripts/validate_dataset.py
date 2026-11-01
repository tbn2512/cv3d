"""Validate a Nerfstudio dataset JSON, image/mask paths, dimensions and splits."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--transforms',default='transforms.json')
    p.add_argument('--require-masks',action='store_true')
    args=p.parse_args()
    meta_path=args.dataset/args.transforms
    if not meta_path.is_file(): raise SystemExit(f'Missing metadata: {meta_path}')
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    frames=meta.get('frames',[])
    errors=[]; warnings=[]
    if not frames: errors.append('No frames in transforms JSON.')
    image_paths=[]; mask_count=0; fg_fracs=[]
    for i,fr in enumerate(frames):
        if 'file_path' not in fr or 'transform_matrix' not in fr:
            errors.append(f'Frame {i} lacks file_path or transform_matrix.'); continue
        image_path=args.dataset/fr['file_path']
        image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        if image is None:
            errors.append(f'Image missing/unreadable: {image_path}'); continue
        image_paths.append(fr['file_path'])
        mat=np.asarray(fr['transform_matrix'],dtype=np.float64)
        if mat.shape!=(4,4) or not np.isfinite(mat).all(): errors.append(f'Invalid 4x4 pose: {fr["file_path"]}')
        if 'w' in meta and int(meta['w'])!=image.shape[1]: warnings.append(f'Image width differs from top-level metadata: {fr["file_path"]}')
        if 'h' in meta and int(meta['h'])!=image.shape[0]: warnings.append(f'Image height differs from top-level metadata: {fr["file_path"]}')
        mp=fr.get('mask_path')
        if args.require_masks and not mp: errors.append(f'Missing mask_path: {fr["file_path"]}')
        if mp:
            mask_path=args.dataset/mp
            mask=cv2.imread(str(mask_path),cv2.IMREAD_UNCHANGED)
            if mask is None: errors.append(f'Mask missing/unreadable: {mask_path}'); continue
            if mask.ndim!=2: errors.append(f'Mask must be one channel: {mask_path}'); continue
            if mask.shape!=image.shape[:2]: errors.append(f'Mask/image size mismatch: {mask_path}'); continue
            vals=set(np.unique(mask).tolist())
            if not vals.issubset({0,255}): errors.append(f'Mask is not binary 0/255: {mask_path}')
            frac=float(np.mean(mask==255)); fg_fracs.append(frac); mask_count+=1
            if frac<0.01 or frac>0.99: warnings.append(f'Foreground fraction suspicious ({frac:.3f}): {mask_path}')
    name_set=set(image_paths)
    for split in ('train','val','test'):
        key=f'{split}_filenames'
        if key in meta:
            missing=set(meta[key])-name_set
            if missing: errors.append(f'{key} contains missing image paths: {sorted(missing)[:5]}')
    if all(k in meta for k in ('train_filenames','val_filenames','test_filenames')):
        a=set(meta['train_filenames']); b=set(meta['val_filenames']); c=set(meta['test_filenames'])
        if a&b or a&c or b&c: errors.append('Train/val/test filename lists overlap.')
        if not c: errors.append('Test split is empty.')
        if not b: errors.append('Validation split is empty.')
    if args.require_masks and mask_count!=len(frames): errors.append(f'Expected {len(frames)} masks, validated {mask_count}.')
    sparse=args.dataset/'colmap'/'sparse'/'0'
    if not sparse.exists(): warnings.append(f'COLMAP sparse model folder not found at {sparse}; inspect ns-process-data output layout.')
    print(f'Frames declared: {len(frames)} | readable images: {len(image_paths)} | masks validated: {mask_count}')
    if fg_fracs: print(f'Mask foreground fraction: median={np.median(fg_fracs):.3f}, min={np.min(fg_fracs):.3f}, max={np.max(fg_fracs):.3f}')
    for w in warnings: print('WARNING:',w)
    for e in errors: print('ERROR:',e)
    if errors: raise SystemExit(f'VALIDATION FAILED ({len(errors)} errors, {len(warnings)} warnings)')
    print(f'VALIDATION PASSED ({len(warnings)} warnings).')

if __name__=='__main__': main()
