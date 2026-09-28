"""
plot_joint_angles.py

Plots the angle of each articulation (joint) stored in a trajectory .npz file
over time, one subplot per joint. Joint limits are read directly from the
MuJoCo model compiled from senecabot_loco.xml (no longer hard-coded): where the
model enforces a limit, the band is drawn and any frames outside it are shaded
red so that unreachable reference motion is immediately visible. Joints that are
unlimited in the model (e.g. the malformed back-leg ranges) are labelled as such.

Usage:
    python plot_joint_angles.py                       # uses trajectory_adapted.npz
    python plot_joint_angles.py path/to/trajectory.npz
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from scipy.signal import find_peaks


def _use_times_new_roman():
    """Render in Times New Roman (vector). The msttcorefonts TTFs ship on this box but
    matplotlib's cache may not have indexed them, so register any found explicitly, then fall
    back to metric-compatible serifs (Nimbus Roman / Liberation Serif) if absent."""
    for p in ("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/timesbi.ttf"):
        if Path(p).exists():
            try:
                _fm.fontManager.addfont(p)
            except Exception:
                pass
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
    })


_use_times_new_roman()

# ------------------------------------------------------------------------------
# PATHS (single source of truth: simulation/config/paths.py; needs `pip install -e .`)
# ------------------------------------------------------------------------------
from simulation.config import paths

OUTPUT_DIR = paths.FIGURES

# Canonical SenecaBot range of motion [degrees], from the design specification table
# (front / back legs). This is the authoritative source for the limit bands — quoted
# directly from the thesis spec, not re-read from any XML.
#   Joint  Front ROM        Back ROM
#   Hip    25 to 105        -105 to -40
#   Knee   34 to 150        -140 to -40
#   Ankle  0 to 160         -85 to 0
ROM_DEG = {
    "fr_hip": (25, 105),   "fl_hip": (25, 105),
    "fr_knee": (34, 150),  "fl_knee": (34, 150),
    "fr_ankle": (0, 160),  "fl_ankle": (0, 160),
    "br_hip": (-105, -40), "bl_hip": (-105, -40),
    "br_knee": (-140, -40), "bl_knee": (-140, -40),
    "br_ankle": (-85, 0),  "bl_ankle": (-85, 0),
}

# Joint nomenclature, in the same order as qpos[:, 7:] (12 hinge joints after
# the 7-DoF free root). Matches the model's qpos layout.
JOINT_NAMES = [
    "fr_hip", "fr_knee", "fr_ankle",
    "fl_hip", "fl_knee", "fl_ankle",
    "br_hip", "br_knee", "br_ankle",
    "bl_hip", "bl_knee", "bl_ankle",
]

def model_joint_ranges() -> dict:
    """Return {joint_name: (lo_rad, hi_rad)} for the 12 joints, from the canonical SenecaBot
    range-of-motion spec (``ROM_DEG``). All 12 joints are limited, so every joint gets a band."""
    return {name: (np.deg2rad(lo), np.deg2rad(hi)) for name, (lo, hi) in ROM_DEG.items()}


def load_qpos(path: Path):
    """Load qpos (N, nq) and sampling frequency from either a Loco-MuJoCo
    Trajectory .npz (e.g. trajectory_adapted.npz) or a plain .npz with a
    'qpos' key (e.g. trajectory.npz / trajectory_augmented.npz)."""
    try:
        from loco_mujoco.trajectory import Trajectory
        traj = Trajectory.load(str(path))
        return np.asarray(traj.data.qpos), float(traj.info.frequency)
    except Exception:
        data = np.load(path, allow_pickle=True)
        qpos = np.asarray(data["qpos"])
        if "time" in data and len(data["time"]) > 1:
            freq = 1.0 / float(np.median(np.diff(np.asarray(data["time"]))))
        else:
            freq = 100.0  # fallback
        return qpos, freq


def joint_limits():
    """Return (lo, hi, limited) arrays in radians for the 12 joints, taken from
    the MuJoCo model's joint limits (see model_joint_ranges)."""
    ranges = model_joint_ranges()
    lo, hi, limited = [], [], []
    for name in JOINT_NAMES:
        if name in ranges:
            a, b = ranges[name]
            lo.append(a); hi.append(b); limited.append(True)
        else:
            lo.append(np.nan); hi.append(np.nan); limited.append(False)
    return np.array(lo), np.array(hi), np.array(limited)


def single_cycle_window(qpos):
    """Return (start, end) frame indices of one gait cycle, detected from FR-hip-angle
    (qpos[:, 7]) troughs — the same boundary the augmentation pipeline uses
    (augment.detect_cycles). One cycle spans trough-to-trough. Falls back to the whole
    trajectory if fewer than two troughs are found."""
    fr_hip = qpos[:, 7]
    troughs, _ = find_peaks(-fr_hip, prominence=0.04, distance=25)
    if len(troughs) >= 2:
        return int(troughs[0]), int(troughs[1])
    return 0, qpos.shape[0]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv            # --full = plot the whole looped clip (old behaviour)
    npz_path = Path(args[0]).resolve() if args else paths.TRAJ_ADAPTED

    qpos, freq = load_qpos(npz_path)

    if full:
        s, e = 0, qpos.shape[0]
        t = np.arange(e - s) / freq                      # seconds
        xlabel, xspan = "time [s]", f"{e - s} frames, {(e - s) / freq:.2f} s"
    else:
        s, e = single_cycle_window(qpos)
        t = np.linspace(0.0, 100.0, e - s)               # percent of gait cycle
        xlabel, xspan = "gait cycle [%]", f"1 cycle = {e - s} frames ({(e - s) / freq:.2f} s)"

    angles = np.rad2deg(qpos[s:e, 7:19])    # (N, 12) joint angles in degrees
    n_steps = angles.shape[0]

    lo, hi, limited = joint_limits()
    lo_deg, hi_deg = np.rad2deg(lo), np.rad2deg(hi)

    print(f"File      : {npz_path.name}")
    print(f"Frames    : {qpos.shape[0]}  @ {freq:.1f} Hz  ({qpos.shape[0] / freq:.2f} s)")
    print(f"Plotting  : {'full clip' if full else 'single gait cycle'}  ({xspan})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    title_span = "full clip" if full else "single gait cycle"
    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
    fig.suptitle(f"Joint angles  —  {npz_path.name}  ({title_span}, {xspan})",
                 fontsize=14)

    for k, ax in enumerate(axes.flat):
        ax.plot(t, angles[:, k], lw=1.4, color="tab:blue")
        ax.set_title(JOINT_NAMES[k])
        ax.grid(True, alpha=0.3)
        if k % 3 == 0:
            ax.set_ylabel("angle [deg]")
        if k >= 9:
            ax.set_xlabel(xlabel)

        if limited[k]:
            # enforced limit band + shading of out-of-range frames
            band_lo, band_hi = sorted((lo_deg[k], hi_deg[k]))
            ax.axhline(band_lo, color="tab:red", ls="--", lw=0.9)
            ax.axhline(band_hi, color="tab:red", ls="--", lw=0.9)
            out = (angles[:, k] < band_lo) | (angles[:, k] > band_hi)
            if out.any():
                ax.fill_between(t, angles[:, k], band_lo,
                                where=angles[:, k] < band_lo,
                                color="red", alpha=0.25)
                ax.fill_between(t, angles[:, k], band_hi,
                                where=angles[:, k] > band_hi,
                                color="red", alpha=0.25)
                pct = 100.0 * out.mean()
                ax.text(0.02, 0.04, f"{pct:.0f}% out of range",
                        transform=ax.transAxes, color="red", fontsize=8)
        else:
            ax.text(0.02, 0.04, "unlimited in model",
                    transform=ax.transAxes, color="gray", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = OUTPUT_DIR / f"joint_angles_{npz_path.stem}.pdf"
    fig.savefig(out_path)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
