"""Merge selected image folders into one joint-SfM folder with unique prefixes.

Example:
 python scripts/merge_capture_images.py --output data/colmap_input/images \
   --group orbit=data/curated/orbit_clean \
   --group spiral=data/curated/spiral_clean \
   --group holdout=data/curated/holdout
"""
from __future__ import annotations
import argparse,shutil
from pathlib import Path
EXTS={'.jpg','.jpeg','.png','.tif','.tiff'}
def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--group',action='append',required=True,help='prefix=folder (repeat)');p.add_argument('--overwrite',action='store_true');a=p.parse_args()
 if a.output.exists() and any(a.output.iterdir()):
  if not a.overwrite:raise SystemExit(f'{a.output} is not empty; use a new directory or --overwrite')
  shutil.rmtree(a.output)
 a.output.mkdir(parents=True,exist_ok=True); total=0
 for item in a.group:
  if '=' not in item:raise SystemExit(f'Bad --group {item}; expected prefix=folder')
  prefix,folder=item.split('=',1); src=Path(folder)
  files=sorted(x for x in src.iterdir() if x.is_file() and x.suffix.lower() in EXTS)
  if not files:raise SystemExit(f'No images in {src}')
  for i,f in enumerate(files,1):
   dst=a.output/f'{prefix}_{i:05d}{f.suffix.lower()}'
   shutil.copy2(f,dst);total+=1
  print(f'{prefix}: {len(files)} images')
 print(f'Merged {total} images into {a.output}')
if __name__=='__main__':main()
