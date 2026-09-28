"""
Central style configuration for all thesis figures.

Why this exists:
    Consistency across figures is the single most visible quality marker
    in a thesis. Every figure goes through `apply_style()` so fonts,
    sizes, line widths, and colors are uniform.

Usage:
    from style import apply_style, StyleConfig, figsize

    apply_style()                          # paper default
    apply_style(StyleConfig(use_latex=True))  # LaTeX-rendered text
    fig, ax = plt.subplots(figsize=figsize("single"))
"""
from dataclasses import dataclass, field
from typing import List, Literal, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Physical sizes
# ----------------------------------------------------------------------
# Adjust these to match your LaTeX template's geometry.
# Run \the\columnwidth and \the\textwidth in your .tex file to get exact pts,
# then divide by 72.27 to convert to inches.
COLUMN_WIDTH_IN = 3.5    # typical single-column (IEEE / two-column papers)
TEXT_WIDTH_IN   = 7.16   # typical full text width
GOLDEN_RATIO    = 1.618


# ----------------------------------------------------------------------
# Color palettes
# ----------------------------------------------------------------------
PALETTES = {
    # Colorblind-friendly (Wong, 2011). Default — works in print and grayscale.
    "wong": ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#F0E442", "#56B4E9", "#E69F00", "#000000"],
    # Muted academic palette
    "muted": ["#4C72B0", "#DD8452", "#55A467", "#C44E52",
              "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"],
    # High-contrast for slides
    "vivid": ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
              "#FF7F00", "#FFFF33", "#A65628", "#F781BF"],
}


# ----------------------------------------------------------------------
# Style preset
# ----------------------------------------------------------------------
@dataclass
class StyleConfig:
    """Reusable style configuration. Override any field to customize."""
    # Typography
    font_family: str = "serif"          # matches LaTeX Computer Modern
    base_font_size: float = 9.0          # body text in pt
    use_latex: bool = False              # True -> use \usepackage{...} renderer
                                         # (slower, needs LaTeX installed)
    # Colors
    palette: str = "wong"

    # Lines / markers
    line_width: float = 1.4
    marker_size: float = 4.0

    # Axes
    grid: bool = True
    grid_alpha: float = 0.3
    spine_top: bool = False              # hide top spine for cleaner look
    spine_right: bool = False

    # Misc
    dpi: int = 300                       # only matters for raster output
    extra_rc: dict = field(default_factory=dict)  # any rcParams override


def apply_style(cfg: StyleConfig | None = None) -> None:
    """Apply the style globally. Call once at the start of every plot script."""
    cfg = cfg or StyleConfig()
    colors = PALETTES[cfg.palette]

    rc = {
        # Fonts
        "font.family":       cfg.font_family,
        "font.size":         cfg.base_font_size,
        "axes.titlesize":    cfg.base_font_size,
        "axes.labelsize":    cfg.base_font_size,
        "xtick.labelsize":   cfg.base_font_size - 1,
        "ytick.labelsize":   cfg.base_font_size - 1,
        "legend.fontsize":   cfg.base_font_size - 1,

        # LaTeX rendering (optional)
        "text.usetex":       cfg.use_latex,
        "pgf.rcfonts":       not cfg.use_latex,

        # Lines & markers
        "lines.linewidth":   cfg.line_width,
        "lines.markersize":  cfg.marker_size,
        "axes.linewidth":    0.8,

        # Color cycle
        "axes.prop_cycle":   mpl.cycler(color=colors),

        # Grid
        "axes.grid":         cfg.grid,
        "grid.alpha":        cfg.grid_alpha,
        "grid.linewidth":    0.5,

        # Spines
        "axes.spines.top":   cfg.spine_top,
        "axes.spines.right": cfg.spine_right,

        # Output
        "figure.dpi":        cfg.dpi,
        "savefig.dpi":       cfg.dpi,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":      42,   # embed TrueType -> editable in Illustrator
        "ps.fonttype":       42,
    }
    rc.update(cfg.extra_rc)
    plt.rcParams.update(rc)


def figsize(width: Literal["single", "double", "full"] = "single",
            ratio: float = 1 / GOLDEN_RATIO,
            height_in: float | None = None) -> Tuple[float, float]:
    """
    Return (width, height) in inches for `plt.subplots(figsize=...)`.

    width:
        "single" -> one column (default)
        "double" -> 2 * column
        "full"   -> full text width
    ratio:
        height / width. Default = 1/phi (golden ratio).
    height_in:
        Override ratio with an explicit height in inches.
    """
    w = {"single": COLUMN_WIDTH_IN,
         "double": 2 * COLUMN_WIDTH_IN,
         "full":   TEXT_WIDTH_IN}[width]
    h = height_in if height_in is not None else w * ratio
    return (w, h)
