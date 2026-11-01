"""Attach masks to a Nerfstudio transforms.json by exact filename stem.

Copies/resizes masks to masks/, masks_2/, masks_4/ matching existing image scales.
Fails on missing matches rather than silently pairing by sorted index.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--source-masks',type=Path,required=True)
    args=p.parse_args()
    meta_path=args.dataset/'transforms.json'
    if not meta_path.is_file(): raise SystemExit(f'Missing {meta_path}')
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    frames=meta.get('frames',[])
    if not frames: raise SystemExit('transforms.json has no frames')
    masks={x.stem:x for x in args.source_masks.iterdir() if x.is_file() and x.suffix.lower()=='.png'}
    missing=[]
    for frame in frames:
        image_rel=Path(frame['file_path'])
        image_path=args.dataset/image_rel
        key=image_path.stem
        if key not in masks: missing.append(key); continue
        mask=cv2.imread(str(masks[key]),cv2.IMREAD_GRAYSCALE)
        image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        if mask is None or image is None: raise SystemExit(f'Unreadable image or mask for {key}')
        # Strict binary single-channel mask; black=ignore, white=keep.
        mask=np.where(mask>=128,255,0).astype(np.uint8)
        for scale,suffix in [(1,''),(2,'_2'),(4,'_4')]:
            image_dir=args.dataset/f'images{suffix}'
            if not image_dir.is_dir(): continue
            scaled_image=image if scale==1 else cv2.imread(str(image_dir/image_rel.name),cv2.IMREAD_COLOR)
            if scaled_image is None: raise SystemExit(f'Missing scaled image: {image_dir/image_rel.name}')
            h,w=scaled_image.shape[:2]
            resized=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
            out_dir=args.dataset/f'masks{suffix}'; out_dir.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(out_dir/(image_rel.stem+'.png')),resized)
        frame['mask_path']=f'masks/{image_rel.stem}.png'
    if missing:
        raise SystemExit('No mask found for these image stems (no changes written): '+', '.join(missing[:20]))
    meta_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'Attached {len(frames)} masks by exact image stem.')

if __name__=='__main__': main()
