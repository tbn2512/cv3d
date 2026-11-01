"""Create a curated copy using a human-reviewed CSV allow-list.

1) Open audit CSV, add a column named keep with true/false values.
2) Run this script. Source images remain untouched.
"""
from __future__ import annotations
import argparse, csv, shutil
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--audit-csv', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--keep-column', default='keep')
    args=p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    copied=0
    with args.audit_csv.open(newline='', encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if args.keep_column not in (reader.fieldnames or []):
            raise SystemExit(f"Add a '{args.keep_column}' column to {args.audit_csv} and mark reviewed frames true/false.")
        for row in reader:
            if row[args.keep_column].strip().lower() in {'1','true','yes','y'}:
                src=Path(row['path'])
                if not src.is_file(): raise SystemExit(f'Missing source image: {src}')
                dst=args.output/src.name
                if dst.exists(): raise SystemExit(f'Duplicate output filename: {dst}')
                shutil.copy2(src,dst); copied+=1
    print(f'Copied {copied} reviewed images to {args.output}; originals unchanged.')

if __name__=='__main__': main()
