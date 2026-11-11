"""Auto-curate candidate frames using sharpness and spacing heuristics.

Reads an audit CSV from audit_images.py, applies condition-aware filtering,
and adds a 'keep' column. Feed the result to select_frames.py.

Usage:
  python scripts/auto_curate.py --audit-csv data/manifests/obj_orbit_audit.csv --condition orbit_clean
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np

# Per-condition configuration
# sharpness_pct: frames below this percentile of Laplacian variance are rejected
# target: desired number of curated frames (uniform subsampling if exceeded)
CONDITION_DEFAULTS = {
    'orbit_clean':   {'sharpness_pct': 15, 'target': 80},
    'spiral_clean':  {'sharpness_pct': 15, 'target': 80},
    'blur_moderate': {'sharpness_pct':  5, 'target': 80},
    'blur_heavy':    {'sharpness_pct':  3, 'target': 80},
    'blur_1_50':     {'sharpness_pct':  5, 'target': 80},
    'blur_1_25':     {'sharpness_pct':  3, 'target': 80},
    'holdout':       {'sharpness_pct': 10, 'target': 42},
}


def main() -> None:
    p = argparse.ArgumentParser(description='Auto-curate candidate frames.')
    p.add_argument('--audit-csv', type=Path, required=True,
                   help='Audit CSV from audit_images.py')
    p.add_argument('--condition', required=True, choices=CONDITION_DEFAULTS,
                   help='Capture condition (determines sharpness threshold and target count)')
    p.add_argument('--target', type=int, default=None,
                   help='Override default target frame count')
    args = p.parse_args()

    cfg = CONDITION_DEFAULTS[args.condition]
    target = args.target or cfg['target']

    # Read audit CSV
    with args.audit_csv.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f'Empty audit CSV: {args.audit_csv}')

    # Ensure fieldnames include 'keep'
    fieldnames = list(rows[0].keys())
    if 'keep' not in fieldnames:
        fieldnames.append('keep')

    # Step 1: mark unreadable frames as rejected
    for r in rows:
        r['keep'] = 'true' if r.get('readable', '').strip().lower() == 'true' else 'false'

    readable_rows = [r for r in rows if r['keep'] == 'true']
    if not readable_rows:
        raise SystemExit('No readable frames found.')

    # Step 2: reject frames below the condition-specific Laplacian variance percentile
    variances = np.array([float(r['laplacian_var']) for r in readable_rows])
    threshold = float(np.percentile(variances, cfg['sharpness_pct']))
    for r in readable_rows:
        if float(r['laplacian_var']) < threshold:
            r['keep'] = 'false'

    kept = [r for r in rows if r['keep'] == 'true']

    # Step 3: uniform subsampling if above target (preserves spatial diversity)
    if len(kept) > target:
        step = len(kept) / target
        keep_paths = {kept[int(i * step)]['path'] for i in range(target)}
        for r in rows:
            if r['keep'] == 'true' and r['path'] not in keep_paths:
                r['keep'] = 'false'

    # Write updated CSV (in-place)
    with args.audit_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    final = sum(1 for r in rows if r['keep'] == 'true')
    print(f'Auto-curate [{args.condition}]: {len(rows)} total -> '
          f'{len(readable_rows)} readable -> {final} kept (target {target})')


if __name__ == '__main__':
    main()
