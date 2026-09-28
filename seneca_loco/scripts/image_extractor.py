"""Extract a single frame from a video (e.g. a wandb Agent Video) as a PNG.

Usage:
    python image_extractor.py VIDEO_PATH [--out NAME.png] [--time SECONDS]

Output goes to artifacts/figures/ (see simulation/config/paths.py).
"""
import argparse
from pathlib import Path

from simulation.config import paths
from simulation.core.utils import extract_frame

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path, help="Path to the source video (.mp4)")
    ap.add_argument("--out", default="frame.png", help="Output file name (saved in artifacts/figures/)")
    ap.add_argument("--time", type=float, default=0.4, help="Timestamp in seconds to extract")
    args = ap.parse_args()

    extract_frame(
        video_path=args.video,
        output_dir=paths.FIGURES,
        file_name=args.out,
        time_seconds=args.time,
    )
    print(f"Saved {paths.FIGURES / args.out}")
