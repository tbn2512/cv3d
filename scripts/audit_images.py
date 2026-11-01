"""Audit image dimensions, readability, Laplacian variance and SHA-256.

Writes a CSV. It never deletes files. Review scores and make a separate curated
folder; there is no universal Laplacian threshold across lenses/resolutions.
"""
from __future__ import annotations
import argparse, csv, hashlib
from pathlib import Path
import cv2
import numpy as np

EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.heic'}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--csv', type=Path, required=True)
    args = p.parse_args()
    files = sorted(x for x in args.input.rglob('*') if x.is_file() and x.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f'No supported images found under {args.input}')
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            rows.append({'path': str(path), 'readable': False, 'width': '', 'height': '', 'laplacian_var': '', 'sha256': ''})
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows.append({'path': str(path), 'readable': True, 'width': img.shape[1], 'height': img.shape[0],
                     'laplacian_var': round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4),
                     'sha256': sha256(path)})
    with args.csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    scores = [r['laplacian_var'] for r in rows if r['readable']]
    print(f'Images: {len(rows)} | readable: {len(scores)} | unreadable: {len(rows)-len(scores)}')
    if scores:
        print(f'Laplacian variance: median={np.median(scores):.2f}, p10={np.percentile(scores,10):.2f}, p90={np.percentile(scores,90):.2f}')
    print(f'Wrote {args.csv}')

if __name__ == '__main__': main()
