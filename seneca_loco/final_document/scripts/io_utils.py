"""
IO helpers: CSV loading and figure saving with strict naming.

Naming convention (enforced by `save_figure`):
    <category>__<descriptor>__<variant>.<ext>

    Examples:
        training__ppo_reward__seed_avg.pdf
        trajectory__hip_flexion__sim_vs_mocap.pdf
        error__tracking_rmse__per_joint.pdf

Why double underscores:
    They survive in LaTeX (single _ is a math operator), they parse cleanly
    in code (just .split("__")), and they make filenames visually scannable.

Every saved figure also logs a line to `images/_manifest.tsv` with
(fig_id, timestamp, source CSVs, plotter, kwargs hash) so you can always
trace a figure back to the data and code that produced it.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


# Project paths — resolved relative to this file so the scripts work
# regardless of the directory from which they are launched.
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMG_DIR  = ROOT / "images"
MANIFEST = IMG_DIR / "_manifest.tsv"

# Allowed categories — extend as your chapters grow.
CATEGORIES = {
    "training",     # learning curves, losses, rewards
    "trajectory",   # joint angles, foot positions over time
    "error",        # tracking error, residuals
    "comparison",   # side-by-side bar / box plots
    "schematic",    # hand-drawn-ish system diagrams
    "diagnostic",   # one-off debugging plots
}


def load_csv(name: str, **read_csv_kwargs) -> pd.DataFrame:
    """Load a CSV from `data/`. Just a thin wrapper that fails loudly."""
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, **read_csv_kwargs)


def build_fig_id(category: str, descriptor: str, variant: str) -> str:
    """Compose and validate a figure id following the naming convention."""
    if category not in CATEGORIES:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Allowed: {sorted(CATEGORIES)}. "
            f"Add new ones to io_utils.CATEGORIES if needed."
        )
    for token in (category, descriptor, variant):
        if not token or "__" in token or " " in token:
            raise ValueError(
                f"Invalid token '{token}': non-empty, no spaces, no '__'."
            )
    return f"{category}__{descriptor}__{variant}"


def save_figure(
    fig: plt.Figure,
    category: str,
    descriptor: str,
    variant: str = "v1",
    *,
    fmt: str = "pdf",
    sources: Iterable[str] = (),
    plotter: str = "",
    kwargs: dict | None = None,
    close: bool = True,
) -> Path:
    """
    Save `fig` to images/<fig_id>.<fmt> and append a manifest entry.

    Parameters
    ----------
    fig
        matplotlib Figure to save.
    category, descriptor, variant
        Pieces of the filename. See module docstring for the convention.
    fmt
        "pdf" (default, vectorial) or "png" for raster-only content.
    sources
        CSVs (or other inputs) used to make this figure. Logged for traceability.
    plotter
        Name of the plotter function used (e.g. "plot_training_curves").
    kwargs
        kwargs passed to the plotter — hashed and logged.
    close
        Close the figure after saving (recommended in batch scripts).
    """
    fig_id = build_fig_id(category, descriptor, variant)
    out = IMG_DIR / f"{fig_id}.{fmt}"
    fig.savefig(out)  # bbox / pad already set globally via rcParams

    # Append manifest row (TSV — easy to diff in git and read in pandas)
    _append_manifest(fig_id=fig_id,
                     out=out,
                     sources=list(sources),
                     plotter=plotter,
                     kwargs=kwargs or {})

    if close:
        plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------
def _append_manifest(*, fig_id: str, out: Path,
                     sources: list[str], plotter: str,
                     kwargs: dict) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.exists():
        MANIFEST.write_text("fig_id\tfile\tplotter\tsources\tkwargs_hash\ttimestamp\n")

    kwargs_json = json.dumps(kwargs, sort_keys=True, default=str)
    kwargs_hash = hashlib.sha1(kwargs_json.encode()).hexdigest()[:10]
    row = "\t".join([
        fig_id,
        out.name,
        plotter,
        ";".join(sources),
        kwargs_hash,
        time.strftime("%Y-%m-%d %H:%M:%S"),
    ])
    with MANIFEST.open("a") as f:
        f.write(row + "\n")
