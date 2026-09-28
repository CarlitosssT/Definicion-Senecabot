"""
Reusable plotter functions. Each one:

  * Takes a DataFrame (or several) plus column names → no hard-coded schemas.
  * Returns a `(fig, ax)` so callers can compose / annotate before saving.
  * Accepts a `style_overrides` dict for one-off rcParams tweaks.

Add new plotters here as new figure types appear in the thesis.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import figsize


# ----------------------------------------------------------------------
# Helper: context manager for one-off style overrides
# ----------------------------------------------------------------------
@contextmanager
def _rc_context(overrides: dict | None):
    if not overrides:
        yield
        return
    with plt.rc_context(overrides):
        yield


# ----------------------------------------------------------------------
# 1. Training curves with multi-seed shading
# ----------------------------------------------------------------------
def plot_training_curves(
    df: pd.DataFrame,
    *,
    x: str = "step",
    y: str = "reward",
    seed_col: str | None = "seed",
    group_col: str | None = None,        # e.g. "method" -> one line per method
    width: str = "single",
    ylabel: str | None = None,
    xlabel: str | None = None,
    smooth_window: int = 1,              # rolling mean window (1 = no smoothing)
    style_overrides: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot training metric over time. Handles:
      * Single run: just (x, y).
      * Multi-seed: mean line + ±1 std shaded band per group.
      * Multi-group: one line per `group_col`, optionally averaged over seeds.
    """
    with _rc_context(style_overrides):
        fig, ax = plt.subplots(figsize=figsize(width))

        groups = [None] if group_col is None else list(df[group_col].unique())

        for g in groups:
            sub = df if g is None else df[df[group_col] == g]
            label = None if g is None else str(g)

            if seed_col and seed_col in sub.columns and sub[seed_col].nunique() > 1:
                # Average over seeds at each x value
                pivot = sub.pivot_table(index=x, columns=seed_col, values=y)
                mean = pivot.mean(axis=1).rolling(smooth_window, min_periods=1).mean()
                std  = pivot.std(axis=1).rolling(smooth_window, min_periods=1).mean()
                line, = ax.plot(mean.index, mean.values, label=label)
                ax.fill_between(mean.index, mean - std, mean + std,
                                alpha=0.18, color=line.get_color(), linewidth=0)
            else:
                ys = sub[y].rolling(smooth_window, min_periods=1).mean()
                ax.plot(sub[x].values, ys.values, label=label)

        ax.set_xlabel(xlabel or x.replace("_", " ").capitalize())
        ax.set_ylabel(ylabel or y.replace("_", " ").capitalize())
        if groups != [None]:
            ax.legend(frameon=False)
        fig.tight_layout()
        return fig, ax


# ----------------------------------------------------------------------
# 2. Multi-DOF trajectory comparison (e.g. sim vs mocap reference)
# ----------------------------------------------------------------------
def plot_trajectory_comparison(
    refs: dict[str, pd.DataFrame],     # {"mocap": df, "sim": df, ...}
    *,
    time_col: str = "t",
    dof_cols: Sequence[str],            # which joints/DOFs to plot
    width: str = "full",
    ncols: int = 3,
    ylabel: str = "Angle [rad]",
    style_overrides: dict | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Grid of subplots, one per DOF. Overlays every DataFrame in `refs` so you
    can visually compare e.g. mocap reference, simulated rollout, and policy
    output on the same axes.
    """
    n = len(dof_cols)
    nrows = (n + ncols - 1) // ncols
    with _rc_context(style_overrides):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize(width, height_in=1.6 * nrows),
            sharex=True,
        )
        axes = np.atleast_2d(axes).reshape(nrows, ncols)

        for k, dof in enumerate(dof_cols):
            ax = axes[k // ncols, k % ncols]
            for label, df in refs.items():
                ax.plot(df[time_col].values, df[dof].values, label=label)
            ax.set_title(dof, pad=2)
            if k // ncols == nrows - 1:
                ax.set_xlabel("Time [s]")
            if k % ncols == 0:
                ax.set_ylabel(ylabel)

        # Turn off unused axes
        for k in range(n, nrows * ncols):
            axes[k // ncols, k % ncols].set_visible(False)

        # One shared legend at the top — tight_layout first, then reserve
        # vertical space so the legend never overlaps subplot titles.
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels,
                   loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.99), frameon=False)
        return fig, axes


# ----------------------------------------------------------------------
# 3. Tracking error over time
# ----------------------------------------------------------------------
def plot_tracking_error(
    df: pd.DataFrame,
    *,
    time_col: str = "t",
    error_cols: Sequence[str],
    width: str = "single",
    ylabel: str = "Error",
    log_y: bool = False,
    style_overrides: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """One axis, one line per `error_cols`. Useful for joint-wise RMSE."""
    with _rc_context(style_overrides):
        fig, ax = plt.subplots(figsize=figsize(width))
        for col in error_cols:
            ax.plot(df[time_col].values, df[col].values, label=col)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig, ax


# ----------------------------------------------------------------------
# 4. Grouped bar comparison (e.g. final metric across methods)
# ----------------------------------------------------------------------
def plot_metric_comparison(
    df: pd.DataFrame,
    *,
    x: str,                              # e.g. "method"
    y: str,                              # e.g. "success_rate"
    err: str | None = None,              # column with std/CI (optional)
    width: str = "single",
    ylabel: str | None = None,
    style_overrides: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    with _rc_context(style_overrides):
        fig, ax = plt.subplots(figsize=figsize(width))
        xs = np.arange(len(df))
        yerr = df[err].values if err else None
        ax.bar(xs, df[y].values, yerr=yerr, capsize=3, width=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(df[x].values, rotation=0)
        ax.set_ylabel(ylabel or y)
        fig.tight_layout()
        return fig, ax


# ----------------------------------------------------------------------
# 5. Schematic: per-leg segment + joint-angle naming convention
# ----------------------------------------------------------------------
def plot_leg_convention(
    _data=None,                          # schematic: ignores any data argument
    *,
    seg_lengths: Sequence[float] = (0.166, 0.231, 0.148),  # upper, lower, foot (hind, m)
    abs_angles_deg: Sequence[float] = (25.0, -12.0, 8.0),  # upper, lower, foot vs. down-vertical
    figsize_in: tuple[float, float] = (4.6, 4.8),
    style_overrides: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Schematic of ONE SenecaBot leg annotated with the project's naming convention.

    The robot is a simplified quadruped: all four legs are identical 3-DOF chains,
    so segments use generic proximal->distal names (upper / lower / foot) and joints
    use functional names (hip / knee / ankle). This figure declares those names plus
    the joint-angle convention: hip is ABSOLUTE (sagittal angle vs. vertical), knee
    and ankle are RELATIVE (inter-segment). Proportions use the real hindlimb segment
    lengths from senecabot_loco.xml; the same names apply to all legs (prefix
    fr / fl / br / bl). Not anatomically literal — see the key for fore/hind anatomy.
    """
    from matplotlib.patches import Arc

    with _rc_context(style_overrides):
        fig, ax = plt.subplots(figsize=figsize_in)

        # --- geometry ---------------------------------------------------
        phi = np.deg2rad(np.asarray(abs_angles_deg))          # vs. down-vertical, +x = forward
        dirs = np.stack([np.sin(phi), -np.cos(phi)], axis=1)   # unit dirs in (x, y)
        pts = np.zeros((4, 2))
        for i, (L, d) in enumerate(zip(seg_lengths, dirs)):
            pts[i + 1] = pts[i] + L * d
        P0, P1, P2, P3 = pts
        seg_dir_deg = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0]))  # vs. +x axis

        seg_colors = ["#3a3a3a", "#5a5a5a", "#7a7a7a"]
        accent = "#D55E00"   # angle arcs (Wong orange)
        ref_col = "#9a9a9a"  # reference / extension dashes

        # --- base / fixed pivot (simple hatched support) ---------------
        xa, ya = P0
        ax.plot([xa - 0.085, xa + 0.085], [ya, ya], color="#333", lw=1.6, zorder=2)
        for xh in np.linspace(xa - 0.075, xa + 0.085, 8):
            ax.plot([xh, xh - 0.026], [ya, ya + 0.03], color="#888", lw=0.8, zorder=1)
        ax.text(xa - 0.105, ya + 0.012, "base", ha="right", va="bottom",
                fontsize=8.5, zorder=3)

        # --- segments ---------------------------------------------------
        seg_names = ["upper", "lower", "foot"]
        seg_anat = ["femur / humerus", "tibia / forearm", "metatarsus / metacarpus"]
        seg_lw = [7.0, 6.0, 5.0]
        for i in range(3):
            a, b = pts[i], pts[i + 1]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=seg_colors[i],
                    lw=seg_lw[i], solid_capstyle="round", zorder=2)
            mid = 0.5 * (a + b)
            lab_x = 0.20
            ax.annotate(
                f"$\\bf{{{seg_names[i]}}}$\n({seg_anat[i]})",
                xy=(mid[0], mid[1]), xytext=(lab_x, mid[1]),
                ha="left", va="center", fontsize=8,
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.7,
                                shrinkA=0, shrinkB=2),
                zorder=4,
            )

        # --- joints -----------------------------------------------------
        for p in (P0, P1, P2):
            ax.plot(*p, "o", ms=9, mfc="white", mec="#222", mew=1.4, zorder=5)
        # foot contact point
        ax.plot(*P3, "o", ms=11, mfc="#C44E52", mec="#222", mew=1.2, zorder=5)
        ax.annotate("foot contact", xy=P3, xytext=(P3[0] + 0.14, P3[1] - 0.01),
                    ha="left", va="center", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="#888", lw=0.7),
                    zorder=4)

        # --- ground -----------------------------------------------------
        gy = P3[1] - 0.012
        ax.plot([P3[0] - 0.18, P3[0] + 0.10], [gy, gy], color="#444", lw=1.0, zorder=1)
        for xg in np.linspace(P3[0] - 0.17, P3[0] + 0.08, 9):
            ax.plot([xg, xg - 0.02], [gy, gy - 0.022], color="#777", lw=0.6, zorder=1)

        # --- angle arcs -------------------------------------------------
        def angle_arc(center, theta1, theta2, r, label, sub, dx=0.0, dy=0.0):
            ax.add_patch(Arc(center, 2 * r, 2 * r, angle=0,
                             theta1=min(theta1, theta2), theta2=max(theta1, theta2),
                             color=accent, lw=2.0, zorder=4))
            mid = np.deg2rad(0.5 * (theta1 + theta2))
            lp = (center[0] + 1.95 * r * np.cos(mid) + dx,
                  center[1] + 1.95 * r * np.sin(mid) + dy)
            ax.text(lp[0], lp[1], label, color=accent, fontsize=14,
                    fontweight="bold", ha="center", va="center", zorder=5)
            ax.text(lp[0], lp[1] - 0.05, sub, color=accent, fontsize=7.5,
                    ha="center", va="center", style="italic", zorder=5)

        # hip: absolute, between down-vertical (-90 deg) and upper segment
        ax.plot([P0[0], P0[0]], [P0[1], P0[1] - 0.13], ls=(0, (4, 3)),
                color=ref_col, lw=1.0, zorder=1)
        angle_arc(P0, -90.0, seg_dir_deg[0], 0.072, r"$\theta_\mathrm{hip}$", "absolute",dx=-0.021,dy=0.030)

        # knee: relative, between extension of upper and lower segment
        ext1 = P1 + 0.085 * dirs[0]
        ax.plot([P1[0], ext1[0]], [P1[1], ext1[1]], ls=(0, (4, 3)),
                color=ref_col, lw=1.0, zorder=1)
        angle_arc(P1, seg_dir_deg[0], seg_dir_deg[1], 0.058, r"$\theta_\mathrm{knee}$", "relative",dx=0.020,dy=0.020)

        # ankle: relative, between extension of lower and foot segment
        ext2 = P2 + 0.075 * dirs[1]
        ax.plot([P2[0], ext2[0]], [P2[1], ext2[1]], ls=(0, (4, 3)),
                color=ref_col, lw=1.0, zorder=1)
        angle_arc(P2, seg_dir_deg[1], seg_dir_deg[2], 0.05, r"$\theta_\mathrm{ankle}$", "relative",
                  dx=-0.050, dy=0.075)

        # --- frame ------------------------------------------------------
        ax.set_aspect("equal")
        ax.set_xlim(-0.27, 0.46)
        ax.set_ylim(-0.60, 0.11)
        ax.axis("off")
        fig.tight_layout()
        return fig, ax


def plot_project_overview(
    _data=None,                          # schematic: ignores any data argument
    *,
    figsize_in: tuple[float, float] = (7.2, 9.4),
    style_overrides: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    General architecture diagram of the SenecaBot DeepMimic project.

    Shows the end-to-end flow — ovine MOCAP -> retargeted reference trajectory ->
    imitation training -> trained policy -> evaluation — and marks (with a star,
    in orange) every component whose behaviour comes from a LOCAL modification to
    the `loco-mujoco-seneca` fork, listed in the side panel. Pure schematic: no
    data input. Coordinates are an abstract 0..100 canvas.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # palette
    C_DATA  = "#0072B2"   # biological data            (Wong blue)
    C_PIPE  = "#009E73"   # seneca_loco pipeline        (Wong green)
    C_TRAIN = "#56B4E9"   # training core              (Wong light blue)
    C_OUT   = "#7a7a7a"   # outputs / evaluation        (grey)
    C_FORK  = "#D55E00"   # local fork modifications    (Wong orange)
    INK     = "#222222"

    with _rc_context(style_overrides):
        fig, ax = plt.subplots(figsize=figsize_in)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("auto")
        ax.axis("off")

        def box(cx, cy, w, h, edge, *, title=None, subtitle=None, body=None,
                fc="white", title_size=8.5, body_size=7.0, lw=1.3, star=False,
                ls="-", title_color=None):
            """Rounded box centred at (cx, cy). Returns (cx, cy, w, h) for routing."""
            p = FancyBboxPatch(
                (cx - w / 2, cy - h / 2), w, h,
                boxstyle="round,pad=0.6,rounding_size=2.4",
                linewidth=lw, edgecolor=edge, facecolor=fc, linestyle=ls,
                mutation_aspect=0.5, zorder=3,
            )
            ax.add_patch(p)
            stacked = bool(body or subtitle)
            ty = cy + h / 2 - 3.0 if stacked else cy
            if title is not None:
                t = (r"$\bigstar$ " if star else "") + title
                ax.text(cx, ty, t, ha="center", va="center" if not stacked else "top",
                        fontsize=title_size, fontweight="bold",
                        color=title_color or edge, zorder=4)
            yb = cy + h / 2 - 7.2
            if subtitle is not None:
                ax.text(cx, ty - 3.4, subtitle, ha="center", va="top",
                        fontsize=6.3, style="italic", color="#555555", zorder=4)
                yb = ty - 7.0
            if body is not None:
                ax.text(cx, yb, body, ha="center", va="top",
                        fontsize=body_size, color=INK, zorder=4, linespacing=1.35)
            return (cx, cy, w, h)

        def chip(cx, cy, w, h, label, *, star=False, fc="#f3faf7", edge=C_PIPE):
            p = FancyBboxPatch(
                (cx - w / 2, cy - h / 2), w, h,
                boxstyle="round,pad=0.4,rounding_size=1.6",
                linewidth=1.0, edgecolor=edge, facecolor=fc,
                mutation_aspect=0.6, zorder=3,
            )
            ax.add_patch(p)
            txt = (r"$\bigstar$" + "\n" if star else "") + label
            ax.text(cx, cy, txt, ha="center", va="center", fontsize=6.3,
                    color=INK, zorder=4, linespacing=1.15)

        def arrow(p_from, p_to, *, color=INK, ls="-", lw=1.4, rad=0.0,
                  shrink=2.0):
            xa, ya, wa, ha = p_from
            xb, yb, wb, hb = p_to
            # connect bottom-of-from to top-of-to when stacked vertically
            if ya > yb:
                start = (xa, ya - ha / 2)
                end   = (xb, yb + hb / 2)
            elif ya < yb:
                start = (xa, ya + ha / 2)
                end   = (xb, yb - hb / 2)
            else:  # horizontal
                if xa < xb:
                    start = (xa + wa / 2, ya); end = (xb - wb / 2, yb)
                else:
                    start = (xa - wa / 2, ya); end = (xb + wb / 2, yb)
            a = FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=11,
                linewidth=lw, color=color, linestyle=ls,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=shrink, shrinkB=shrink, zorder=2,
            )
            ax.add_patch(a)

        # ---- title -----------------------------------------------------
        ax.text(50, 99.0, "SenecaBot — DeepMimic imitation-learning pipeline",
                ha="center", va="top", fontsize=11, fontweight="bold", color=INK)

        cx = 33          # main spine x-centre
        W  = 60          # main box width

        # ---- S1: biological data --------------------------------------
        s1 = box(cx, 92.5, W, 6.0, C_DATA, fc="#eaf3fb",
                 title="Ovine (sheep) motion capture  —  C3D markers @ 200 Hz")

        # ---- S2: MOCAP -> reference trajectory pipeline ----------------
        s2 = box(cx, 76.5, W, 19.0, C_PIPE, fc="#f3faf7")
        ax.text(cx, 84.6, "MOCAP → reference-trajectory pipeline", ha="center",
                va="top", fontsize=8.5, fontweight="bold", color=C_PIPE, zorder=4)
        ax.text(cx, 81.7, "seneca_loco / simulation / mocap", ha="center", va="top",
                fontsize=6.2, style="italic", color="#557755", zorder=4)
        # sub-step chips
        chip_y = 77.0
        chips = [("io.py\nC3D→markers", 9.0),
                 ("kinematics\nmarkers→angles", 22.0),
                 ("build.py\nFK→qpos/qvel", 34.5),
                 ("cyclic.py\ncycle+symm+tile", 47.5),
                 ("traj_gen\nsite FK", 58.0)]
        prev = None
        for lab, xx in chips:
            chip(xx, chip_y, 11.0, 7.0, lab)
            cur = (xx, chip_y, 11.0, 7.0)
            if prev is not None:
                arrow(prev, cur, color=C_PIPE, lw=1.1)
            prev = cur
        # output line
        ax.text(cx, 70.6,
                "trajectory_adapted.npz  —  root pose + 12 joint trajectories\n"
                "+ “*_mimic” site targets, resampled to the control rate",
                ha="center", va="top", fontsize=6.6, style="italic", color=INK,
                zorder=4, linespacing=1.3)

        # ---- S3: robot model ------------------------------------------
        s3 = box(cx, 60.5, W, 6.4, C_OUT, fc="#f2f2f2",
                 title="Robot model  —  senecabot_loco.xml",
                 body="12 DoF (4×3 hinges, sagittal plane)  ·  direct-torque motors (DefaultControl)",
                 title_size=8.0, body_size=6.4)

        # ---- S4: imitation training -----------------------------------
        s4 = box(cx, 42.5, W, 23.0, C_TRAIN, fc="#eaf6fd",
                 title="Imitation training — Hydra + PPO",
                 subtitle="seneca_loco/.../train.py + conf.yaml  ·  loco-mujoco-seneca (fork)",
                 title_size=8.5)
        # training component chips (★ = behaviour from a local fork change)
        tchips = [
            ("ImitationFactory", 12.5, 43.5, False),
            ("TrajectoryHandler\n(RSI)", 28.0, 43.5, False),
            ("GoalTrajMimic", 43.0, 43.5, False),
            ("MimicReward", 16.5, 36.0, True),
            ("PPOJax", 33.0, 36.0, True),
            ("MetricsHandler\n(DTW / Fréchet)", 49.5, 36.0, False),
        ]
        for lab, xx, yy, st in tchips:
            chip(xx, yy, 13.5, 6.4, lab, star=st,
                 fc=("#fdeee4" if st else "#eef7fd"),
                 edge=(C_FORK if st else C_TRAIN))

        # ---- S5: trained policy ---------------------------------------
        s5 = box(cx, 27.0, W, 6.0, C_OUT, fc="#f2f2f2",
                 title="Trained policy  —  PPOJax_saved.pkl  (best-validation checkpoint)",
                 title_size=8.0)

        # ---- S6: evaluation -------------------------------------------
        s6 = box(cx, 16.5, W, 8.2, C_OUT, fc="#f2f2f2",
                 title="Evaluation & diagnostics",
                 body="eval.py (replay)  ·  debug_agent.py (figures)\n"
                      "agent_diagnostics.py (speed_ratio, RMS, composite)",
                 title_size=8.0, body_size=6.5)

        # ---- main-spine arrows ----------------------------------------
        arrow(s1, s2, color=INK)
        arrow(s2, s3, color=INK)
        arrow(s3, s4, color=INK)
        arrow(s4, s5, color=INK)
        arrow(s5, s6, color=INK)

        # ---- side panel: local fork modifications ---------------------
        px, pw = 82.0, 32.0
        pcy, ph = 60.0, 46.0
        panel = FancyBboxPatch(
            (px - pw / 2, pcy - ph / 2), pw, ph,
            boxstyle="round,pad=0.8,rounding_size=2.4",
            linewidth=1.6, edgecolor=C_FORK, facecolor="#fff6f0",
            mutation_aspect=0.5, zorder=3,
        )
        ax.add_patch(panel)
        ax.text(px, pcy + ph / 2 - 3.0,
                r"$\bigstar$  Local fork modifications" + "\n(loco-mujoco-seneca)",
                ha="center", va="top", fontsize=8.2, fontweight="bold",
                color=C_FORK, zorder=4, linespacing=1.25)
        fork_lines = (
            r"$\bf{MimicReward}$ (reward/trajectory_based.py)",
            "  · rootvel — forward-velocity",
            "    tracking term",
            "  · track_root_xy=false (drop",
            "    unobservable root x,y)",
            "  · qvel_w_exp made tunable",
            "",
            r"$\bf{PPOJax}$ (algorithms/ppo_jax.py)",
            "  · live wandb logging",
            "    (io_callback)",
            "  · best-checkpoint buffer",
            "  · live validation streaming",
            "",
            r"$\bf{MujocoViewer}$ (visuals)",
            "  · record_camera — video from",
            "    a named XML camera",
            "",
            "Additive & opt-in;",
            "no change to PPO math.",
        )
        ax.text(px - pw / 2 + 2.4, pcy + ph / 2 - 9.5, "\n".join(fork_lines),
                ha="left", va="top", fontsize=6.2, color=INK, zorder=4,
                linespacing=1.3)
        # dashed connector: fork panel -> training box
        a = FancyArrowPatch(
            (px - pw / 2, 42.5), (cx + W / 2, 42.5),
            arrowstyle="-|>", mutation_scale=11, linewidth=1.3,
            color=C_FORK, linestyle=(0, (4, 2)),
            connectionstyle="arc3,rad=0.0", shrinkA=3, shrinkB=3, zorder=2,
        )
        ax.add_patch(a)

        fig.tight_layout(pad=0.4)
        return fig, ax


# ----------------------------------------------------------------------
# 8. Validation tracking distances over training (wandb export)
# ----------------------------------------------------------------------
# The headline run (5_best_try / faithful-sunset-317) was exported from wandb as one
# CSV per (measure, quantity) into data/Wandb_stats_5_best_try/. Each CSV has columns
# "Step", "<run> - Validation Measures/<measure>/<quantity>" (+ __MIN/__MAX duplicates
# for a single run). The value-column header is the canonical metric path, so series
# are keyed off it (robust to the few mislabelled filenames).
_WANDB_FOLDER = "Wandb_stats_5_best_try"
_VD_MEASURES = [("euclidean_distance", "Euclidean"),
                ("dynamic_time_warping", "DTW"),
                ("discrete_frechet_distance", "Discrete Fréchet")]
_VD_QUANTS = [("qpos", "JointPosition"), ("qvel", "JointVelocity"),
              ("site_rpos", "RelSitePosition"), ("site_rrotvec", "RelSiteOrientation"),
              ("site_rvel", "RelSiteVelocity")]


def _load_wandb_series(folder: str) -> dict:
    """Return {(measure, quantity): (steps, values)} parsed from every CSV in
    data/<folder>/, identifying each series by its value-column header."""
    import glob
    from io_utils import DATA_DIR
    out = {}
    for path in sorted(glob.glob(str(DATA_DIR / folder / "*.csv"))):
        df = pd.read_csv(path)
        valcols = [c for c in df.columns
                   if c != "Step" and not c.endswith("__MIN") and not c.endswith("__MAX")]
        if not valcols or "Validation Measures/" not in valcols[0]:
            continue
        col = valcols[0]
        measure, quantity = col.split("Validation Measures/")[1].split("/")
        s = df[["Step", col]].dropna().sort_values("Step")
        out[(measure, quantity)] = (s["Step"].to_numpy() / 1e6, s[col].to_numpy())
    return out


def plot_validation_distance_grid(
    _data=None,                          # reads the wandb folder directly; ignores dispatched data
    *,
    folder: str = _WANDB_FOLDER,
    figsize_in: tuple[float, float] = (7.6, 6.2),
    tick_step_m: float = 100,            # x-axis tick spacing [millions of steps]
    style_overrides: dict | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """3x5 grid of the 15 validation tracking distances vs. env-steps (rows = distance
    measure, cols = tracked quantity), each panel on its own y-scale. The x-axis is ticked
    every ``tick_step_m`` million steps (0, 100, 200, 300 M); each y-axis carries the
    tracked quantity's unit (qpos/qvel are mixed because they include the free-root DoFs)."""
    from matplotlib.ticker import MultipleLocator
    # Distance unit = the tracked quantity's own unit. JointPosition/Velocity span the
    # free root (translation in m, m/s) AND the hinge joints (rad, rad/s), so they are mixed.
    units = {"qpos": "rad, m", "qvel": "rad/s, m/s",
             "site_rpos": "m", "site_rrotvec": "rad", "site_rvel": "m/s"}
    series = _load_wandb_series(folder)
    with _rc_context(style_overrides):
        fig, axes = plt.subplots(len(_VD_MEASURES), len(_VD_QUANTS),
                                 figsize=figsize_in, sharex=True)
        for r, (mk, ml) in enumerate(_VD_MEASURES):
            for c, (qk, ql) in enumerate(_VD_QUANTS):
                ax = axes[r, c]
                if (mk, qk) in series:
                    st, va = series[(mk, qk)]
                    ax.plot(st, va, lw=1.2, marker="o", ms=2.6)
                ax.xaxis.set_major_locator(MultipleLocator(tick_step_m))
                if r == 0:
                    ax.set_title(ql, fontsize=8)
                # y-axis unit on every panel; the leftmost column also names the measure
                if c == 0:
                    ax.set_ylabel(f"{ml}\n[{units[qk]}]", fontsize=7.5)
                else:
                    ax.set_ylabel(f"[{units[qk]}]", fontsize=6.8)
                if r == len(_VD_MEASURES) - 1:
                    ax.set_xlabel("env-steps [M]", fontsize=7)
                ax.tick_params(labelsize=6.0)
                ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig, axes


def plot_entropy_curve(
    _data=None,
    *,
    folder: str = _WANDB_FOLDER,
    fname: str = "entropy.csv",
    figsize_in: tuple[float, float] = (5.2, 3.2),
    style_overrides: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Policy entropy (Train/Entropy) vs. env-steps for the trained agent."""
    from io_utils import DATA_DIR
    df = pd.read_csv(DATA_DIR / folder / fname)
    col = [c for c in df.columns
           if c != "Step" and not c.endswith("__MIN") and not c.endswith("__MAX")][0]
    s = df[["Step", col]].dropna().sort_values("Step")
    with _rc_context(style_overrides):
        fig, ax = plt.subplots(figsize=figsize_in)
        ax.plot(s["Step"].to_numpy() / 1e6, s[col].to_numpy(), lw=1.3)
        ax.set_xlabel("env-steps [M]", fontsize=11)
        ax.set_ylabel("policy entropy", fontsize=11)
        ax.set_title("Policy entropy over training", fontsize=12.5)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig, ax


# ----------------------------------------------------------------------
# 9. Episode return / length and the validation "Metric for Sweep" (wandb export)
# ----------------------------------------------------------------------
def _read_wandb_metric(folder: str, fname: str):
    """(steps in millions, values) from one wandb-export CSV (value = the non-MIN/MAX column)."""
    from io_utils import DATA_DIR
    df = pd.read_csv(DATA_DIR / folder / fname)
    col = [c for c in df.columns
           if c != "Step" and not c.endswith("__MIN") and not c.endswith("__MAX")][0]
    s = df[["Step", col]].dropna().sort_values("Step")
    return s["Step"].to_numpy() / 1e6, s[col].to_numpy()


def plot_episode_return_length(
    _data=None,
    *,
    folder: str = _WANDB_FOLDER,
    figsize_in: tuple[float, float] = (7.16, 3.2),
    tick_step_m: float = 100,
    style_overrides: dict | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Two training-progress monitors of the trained agent vs. env-steps: mean episode return
    (left) and mean episode length (right)."""
    from matplotlib.ticker import MultipleLocator
    panels = [("MeanEpisodeReturn.csv", "mean episode return", "Mean episode return"),
              ("MeanEpisodeLength.csv", "mean episode length [steps]", "Mean episode length")]
    with _rc_context(style_overrides):
        fig, axes = plt.subplots(1, 2, figsize=figsize_in)
        for ax, (fname, ylabel, title) in zip(axes, panels):
            st, va = _read_wandb_metric(folder, fname)
            ax.plot(st, va, lw=1.3)
            ax.set_xlabel("env-steps [M]")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.xaxis.set_major_locator(MultipleLocator(tick_step_m))
            ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig, axes


def plot_metric_for_sweep(
    _data=None,
    *,
    folder: str = _WANDB_FOLDER,
    fname: str = "Metric_for_sweep.csv",
    figsize_in: tuple[float, float] = (5.4, 3.3),
    tick_step_m: float = 100,
    style_overrides: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """The validation 'Metric for Sweep' (Euclidean distance on the relative-site kinematics
    site_rpos+site_rrotvec+site_rvel — the PPO/Optuna objective) vs. env-steps. Logged only on
    validation updates, hence the sparse markers."""
    from matplotlib.ticker import MultipleLocator
    st, va = _read_wandb_metric(folder, fname)
    with _rc_context(style_overrides):
        fig, ax = plt.subplots(figsize=figsize_in)
        ax.plot(st, va, lw=1.3, marker="o", ms=3.5)
        ax.set_xlabel("env-steps [M]", fontsize=11)
        ax.set_ylabel("Metric for Sweep\n(Euclidean, rel-site kinematics)", fontsize=11)
        ax.set_title("Validation objective over training", fontsize=12.5)
        ax.tick_params(labelsize=10)
        ax.xaxis.set_major_locator(MultipleLocator(tick_step_m))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig, ax


def plot_wandb_metric_curve(
    _data=None,
    *,
    folder: str = _WANDB_FOLDER,
    fname: str,
    ylabel: str,
    title: str,
    marker: bool = False,
    figsize_in: tuple[float, float] = (5.4, 3.3),
    tick_step_m: float = 100,
    style_overrides: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Generic single-metric wandb-export curve vs. env-steps (used for episode return / length)."""
    from matplotlib.ticker import MultipleLocator
    st, va = _read_wandb_metric(folder, fname)
    with _rc_context(style_overrides):
        fig, ax = plt.subplots(figsize=figsize_in)
        ax.plot(st, va, lw=1.3, **(dict(marker="o", ms=3.5) if marker else {}))
        ax.set_xlabel("env-steps [M]", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12.5)
        ax.tick_params(labelsize=10)
        ax.xaxis.set_major_locator(MultipleLocator(tick_step_m))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig, ax
