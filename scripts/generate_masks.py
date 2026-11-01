"""Generate draft foreground masks with rembg. Every mask needs visual QC.

Install optional dependencies: pip install rembg onnxruntime
For GPU ONNX, use a compatible onnxruntime-gpu build for your CUDA/cuDNN stack;
CPU onnxruntime is the safer baseline and often sufficient for a few hundred frames.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from rembg import new_session, remove

EXTS={'.jpg','.jpeg','.png','.tif','.tiff'}

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='isnet-general-use')
    p.add_argument('--threshold',type=int,default=127)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    session=new_session(args.model)
    files=sorted(x for x in args.input.iterdir() if x.is_file() and x.suffix.lower() in EXTS)
    if not files: raise SystemExit(f'No images found in {args.input}')
    for i,path in enumerate(files,1):
        with Image.open(path) as im:
            rgba=remove(im.convert('RGB'),session=session).convert('RGBA')
            alpha=np.asarray(rgba.getchannel('A'))
        mask=np.where(alpha>=args.threshold,255,0).astype(np.uint8)
        Image.fromarray(mask,mode='L').save(args.output/(path.stem+'.png'))
        if i%25==0 or i==len(files): print(f'{i}/{len(files)}')
    print(f'Draft masks written to {args.output}. Inspect and correct them before training.')

if __name__=='__main__': main()
