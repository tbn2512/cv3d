from __future__ import annotations
import argparse,csv,hashlib
from pathlib import Path

def digest(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); a=ap.parse_args()
    bad=[]; checked=0
    with a.manifest.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p=a.root/row['relative_path']; checked+=1
            if not p.is_file() or digest(p)!=row['sha256']: bad.append(row['relative_path'])
    print(f'Checked {checked} files; failures={len(bad)}')
    for x in bad[:30]: print('MISMATCH/MISSING:',x)
    if bad: raise SystemExit(1)
    print('All checksums match.')
if __name__=='__main__':main()
