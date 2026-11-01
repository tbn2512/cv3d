"""Extract a reproducible candidate frame set from a video using FFmpeg.

Example:
  python scripts/extract_video_frames.py --video data/raw/orbit_clean.mov \
      --output data/candidates/orbit_clean --fps 2

This does not select or delete frames. It creates PNGs from decoded video frames.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fps', type=float, default=2.0)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if not args.video.is_file():
        raise SystemExit(f'Video not found: {args.video}')
    if args.fps <= 0:
        raise SystemExit('--fps must be positive')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise SystemExit('ffmpeg was not found on PATH. Install FFmpeg and reopen PowerShell.')
    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            raise SystemExit(f'Output is not empty: {args.output}. Choose a new folder or pass --overwrite.')
        for f in args.output.glob('frame_*.png'):
            f.unlink()
    args.output.mkdir(parents=True, exist_ok=True)
    pattern = args.output / 'frame_%05d.png'
    cmd = [ffmpeg, '-hide_banner', '-y' if args.overwrite else '-n', '-i', str(args.video),
           '-vf', f'fps={args.fps}', '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24', '-compression_level', '3', str(pattern)]
    print('Running:', subprocess.list2cmdline(cmd))
    subprocess.run(cmd, check=True)
    count = len(list(args.output.glob('frame_*.png')))
    print(f'Extracted {count} frames into {args.output}')

if __name__ == '__main__':
    main()
