"""Write a relative-path SHA-256 manifest for a directory tree."""
from __future__ import annotations
import argparse, csv, hashlib
from pathlib import Path

def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); files=sorted(x for x in a.root.rglob('*') if x.is_file() and x.resolve()!=a.output.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['relative_path','size_bytes','sha256'])
        for x in files: w.writerow([x.relative_to(a.root).as_posix(),x.stat().st_size,digest(x)])
    print(f'Wrote checksums for {len(files)} files to {a.output}')

if __name__=='__main__': main()
