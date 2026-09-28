"""
Generate every figure in the registry.

Usage:
    python scripts/generate_all.py                # all figures
    python scripts/generate_all.py training__*    # filter by glob on fig_id
    python scripts/generate_all.py --latex        # use LaTeX text renderer
    python scripts/generate_all.py --fmt png      # raster output

Run from any directory — paths resolve via scripts/io_utils.py.
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

# Make scripts/ importable when launched as `python scripts/generate_all.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from io_utils import IMG_DIR, MANIFEST, load_csv, save_figure
from registry import FIGURES, check_unique, get_plotter
from style import StyleConfig, apply_style


def _load_sources(sources: tuple[str, ...]) -> list[pd.DataFrame]:
    return [load_csv(s) for s in sources]


def _dispatch(spec) -> tuple:
    """Call the plotter with the right argument shape for its signature."""
    fn = get_plotter(spec.plotter)
    dfs = _load_sources(spec.sources)

    # Convention:
    #   * 1 source → pass the DataFrame positionally.
    #   * >1 source → pass a {filename_stem: df} dict (for comparison plots).
    if len(dfs) == 1:
        return fn(dfs[0], **spec.kwargs)
    labels = [Path(s).stem.split("_")[0] for s in spec.sources]
    return fn(dict(zip(labels, dfs)), **spec.kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="*",
                        help="Optional glob(s) on fig_id to filter.")
    parser.add_argument("--latex", action="store_true",
                        help="Render text with LaTeX (needs LaTeX installed).")
    parser.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"],
                        help="Output format (default: pdf).")
    parser.add_argument("--clean", action="store_true",
                        help="Delete the images/ folder first.")
    args = parser.parse_args()

    if args.clean and IMG_DIR.exists():
        for p in IMG_DIR.glob("*"):
            p.unlink()
        print(f"[clean] wiped {IMG_DIR}")

    check_unique()
    apply_style(StyleConfig(use_latex=args.latex))

    selected = FIGURES
    if args.patterns:
        selected = [s for s in FIGURES
                    if any(fnmatch.fnmatch(s.fig_id, p) for p in args.patterns)]

    print(f"Generating {len(selected)} figure(s) → {IMG_DIR}")
    for spec in selected:
        try:
            fig, _ = _dispatch(spec)
        except FileNotFoundError as e:
            print(f"  [skip] {spec.fig_id}: {e}")
            continue
        out = save_figure(
            fig,
            category=spec.category,
            descriptor=spec.descriptor,
            variant=spec.variant,
            fmt=args.fmt,
            sources=spec.sources,
            plotter=spec.plotter,
            kwargs=spec.kwargs,
        )
        print(f"  [ok]   {out.name}")

    print(f"\nManifest: {MANIFEST}")


if __name__ == "__main__":
    main()
