"""
gait_symmetry.py — bilateral (left-right) gait-symmetry metrics for a processed trajectory.

A quadruped walk/trot is bilaterally symmetric: a contralateral limb performs the SAME joint
motion shifted by ~half a stride. This script quantifies that with two complementary measures,
computed on the gait-cycle-folded joint angles (cycle = FR-hip-trough to FR-hip-trough):

  1. Phase-shifted symmetry index (headline).
       For each contralateral joint pair (fr<->fl, br<->bl) x {hip,knee,ankle}:
         - best-fit contralateral lag tau*  (argmin RMS over circular phase shifts; ~50% if
           symmetric)  -> a TIMING-symmetry read,
         - residual RMS at that lag and at the fixed 50% shift, normalised by the pair's mean
           range of motion (ROM)                                  -> a WAVEFORM-symmetry read,
         - Pearson correlation of right vs. half-shifted left.
  2. Robinson symmetry index (SI) on scalar features (ROM, peak, mean) per pair:
         SI = |X_R - X_L| / (0.5 (X_R + X_L)) * 100 %   (0 % = perfect symmetry).

Outputs a printed table + a per-pair overlay figure (right vs. half-cycle-shifted left) to
artifacts/figures/thesis_eval/gait_symmetry.{pdf,png}.

Run from the seneca_loco repo root:
    python -m simulation.analysis.gait_symmetry                       # trajectory_adapted.npz
    python -m simulation.analysis.gait_symmetry --npz path/to/traj.npz
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")          # only needed for --agent (env build); harmless otherwise

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from scipy.signal import find_peaks

from simulation.config import paths
from simulation.analysis.plot_joint_angles import load_qpos

OUT_DIR = paths.FIGURES / "thesis_eval"
N_PHASE = 100                                  # gait-cycle resampling resolution (1 sample = 1%)

# Contralateral joint pairs, indices into qpos[:, 7:19] (12 hinge joints):
#   0 fr_hip 1 fr_knee 2 fr_ankle | 3 fl_hip 4 fl_knee 5 fl_ankle
#   6 br_hip 7 br_knee 8 br_ankle | 9 bl_hip 10 bl_knee 11 bl_ankle
PAIRS = [   # (joint type, fore/hind, right index, left index)
    ("hip",   "fore", 0, 3), ("knee", "fore", 1, 4), ("ankle", "fore", 2, 5),
    ("hip",   "hind", 6, 9), ("knee", "hind", 7, 10), ("ankle", "hind", 8, 11),
]


def _use_times_new_roman():
    for p in ("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf"):
        if Path(p).exists():
            try:
                _fm.fontManager.addfont(p)
            except Exception:
                pass
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42,
    })


# ----------------------------------------------------------------------
# gait-cycle folding
# ----------------------------------------------------------------------
def fold_mean_cycle(qpos, n_phase=N_PHASE):
    """Fold a (N, nq) qpos trajectory into ONE mean gait cycle of the 12 joint angles, in degrees.
    Cycle boundaries = FR-hip (qpos[:,7]) troughs. Each cycle is resampled to ``n_phase`` points
    on a circular phase grid [0,1) (endpoint excluded so a half-cycle shift is a clean np.roll),
    then averaged across cycles. Returns (mean_cycle (n_phase,12) [deg], n_cycles)."""
    fr_hip = qpos[:, 7]
    prom = max(float(np.std(fr_hip)) * 0.4, 0.05)
    troughs, _ = find_peaks(-fr_hip, prominence=prom, distance=20)
    ang = np.rad2deg(qpos[:, 7:19])
    xp = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    cycles = []
    for a, b in zip(troughs[:-1], troughs[1:]):
        if b - a < 10:
            continue
        src = np.linspace(0.0, 1.0, b - a)
        cycles.append(np.stack([np.interp(xp, src, ang[a:b, k]) for k in range(12)], axis=1))
    if len(cycles) < 1:
        raise RuntimeError("no full gait cycle detected (need >=2 FR-hip troughs)")
    return np.mean(cycles, axis=0), len(cycles)


# ----------------------------------------------------------------------
# symmetry metrics
# ----------------------------------------------------------------------
def _best_lag(qr, ql):
    """Circular phase lag s (samples) minimising RMS(qr - roll(ql, -s)); returns (s, rms_at_s)."""
    n = len(ql)
    rms = np.array([np.sqrt(np.mean((qr - np.roll(ql, -s)) ** 2)) for s in range(n)])
    s = int(np.argmin(rms))
    return s, float(rms[s])


def _robinson_si(xr, xl):
    denom = 0.5 * (abs(xr) + abs(xl))
    return 100.0 * abs(xr - xl) / denom if denom > 1e-9 else 0.0


def symmetry_table(mean_cycle):
    """Per-pair symmetry metrics from a (n_phase, 12) mean gait cycle [deg]."""
    n = mean_cycle.shape[0]
    half = n // 2
    rows = []
    for jtype, side, ri, li in PAIRS:
        qr, ql = mean_cycle[:, ri], mean_cycle[:, li]
        rom = 0.5 * (np.ptp(qr) + np.ptp(ql))                  # mean ROM of the pair
        # fixed 50% shift
        rms50 = np.sqrt(np.mean((qr - np.roll(ql, -half)) ** 2))
        # best-fit lag
        s_star, rms_star = _best_lag(qr, ql)
        tau_star = 100.0 * s_star / n
        ql_shift = np.roll(ql, -half)
        corr = float(np.corrcoef(qr, ql_shift)[0, 1]) if np.ptp(qr) > 1e-6 else np.nan
        rows.append(dict(
            joint=f"{side}_{jtype}", ri=ri, li=li, rom=rom,
            rms50_deg=rms50, gsi50=rms50 / rom if rom > 1e-6 else np.nan,
            tau_star=tau_star, rms_star_deg=rms_star,
            gsi_star=rms_star / rom if rom > 1e-6 else np.nan, corr=corr,
            si_rom=_robinson_si(np.ptp(qr), np.ptp(ql)),
            si_peak=_robinson_si(float(np.max(qr)), float(np.max(ql))),
            si_mean=_robinson_si(float(np.mean(qr)), float(np.mean(ql))),
        ))
    return rows


def print_table(rows, n_cycles):
    print(f"\nBilateral gait symmetry  (mean of {n_cycles} cycles; lower GSI / SI = more symmetric)\n")
    hdr = (f"{'pair':<12}{'ROM°':>7}{'RMS@50°':>9}{'GSI@50':>8}"
           f"{'tau*%':>7}{'GSI@tau*':>9}{'corr':>7}{'SI_ROM%':>9}{'SI_peak%':>9}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['joint']:<12}{r['rom']:>7.1f}{r['rms50_deg']:>9.2f}{r['gsi50']:>8.3f}"
              f"{r['tau_star']:>7.0f}{r['gsi_star']:>9.3f}{r['corr']:>7.2f}"
              f"{r['si_rom']:>9.1f}{r['si_peak']:>9.1f}")
    g50 = np.nanmean([r["gsi50"] for r in rows])
    gstar = np.nanmean([r["gsi_star"] for r in rows])
    fore = np.nanmean([r["gsi50"] for r in rows if r["joint"].startswith("fore")])
    hind = np.nanmean([r["gsi50"] for r in rows if r["joint"].startswith("hind")])
    print("-" * len(hdr))
    print(f"aggregate   GSI@50 = {g50:.3f}   ({100*(1-g50):.1f}% symmetric)   |   "
          f"GSI@tau* = {gstar:.3f}")
    print(f"            fore GSI@50 = {fore:.3f}   hind GSI@50 = {hind:.3f}")
    print(f"            mean best-fit lag tau* = {np.mean([r['tau_star'] for r in rows]):.0f}% "
          f"(50% expected for a symmetric gait)")


# ----------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------
def plot_overlay(mean_cycle, rows, out):
    n = mean_cycle.shape[0]
    half = n // 2
    x = np.linspace(0, 100, n, endpoint=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    titles = {0: "fore", 1: "hind"}
    for idx, (r, ax) in enumerate(zip(rows, axes.flat)):
        qr = mean_cycle[:, r["ri"]]
        ql_shift = np.roll(mean_cycle[:, r["li"]], -half)
        ax.plot(x, qr, color="tab:blue", lw=1.8, label="right limb")
        ax.plot(x, ql_shift, color="tab:red", lw=1.8, ls="--",
                label="left limb (shifted +50%)")
        ax.set_title(f"{r['joint']}   GSI@50={r['gsi50']:.3f}  (corr {r['corr']:.2f})",
                     fontsize=12)
        ax.grid(True, alpha=0.3); ax.tick_params(labelsize=10)
        if r["ri"] in (0, 6):
            ax.set_ylabel("angle [deg]", fontsize=11)
        if idx >= 3:
            ax.set_xlabel("% gait cycle", fontsize=11)
    axes[0, 0].legend(fontsize=9, loc="best")
    fig.suptitle("Bilateral gait symmetry — right limb vs. half-cycle-shifted left limb",
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=150)
    plt.close(fig)
    print(f"\nwrote {out.name}.pdf / .png  in {OUT_DIR}")


def agent_qpos(path, n_steps=400, seed=0):
    """Roll out the trained policy and return its (N, nq) joint trajectory for a single
    representative (surviving) env, with RSI OFF so all envs share one phase and the cycle fold
    keeps the true joint amplitudes (see CLAUDE.md 2026-06-20 'later 13')."""
    from simulation.analysis.thesis_eval_figures import build_env
    from simulation.analysis.debug_agent import rollout
    env, ac, as_, cfg = build_env(path)
    env.th.random_start = False                                # turn OFF RSI
    d = rollout(env, ac, as_, 8, n_steps, deterministic=True, seed=seed)
    done = np.asarray(d["done"]).astype(bool)
    surv = ~done.any(axis=0)
    env_i = int(np.argmax(surv)) if surv.any() else 0
    return np.asarray(d["qpos"])[:, env_i, :], float(env.dt)


def plot_compare(rows_ref, rows_agent, out):
    """Grouped-bar comparison of the per-joint symmetry index, reference vs. trained agent."""
    labels = [r["joint"] for r in rows_ref]
    gref = [r["gsi50"] for r in rows_ref]
    gag = [r["gsi50"] for r in rows_agent]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - w / 2, gref, w, color="tab:orange", label="reference")
    ax.bar(x + w / 2, gag, w, color="tab:blue", label="trained agent")
    for xi, v in zip(x - w / 2, gref):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w / 2, gag):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("GSI@50  (0 = perfect symmetry)", fontsize=12)
    ax.set_title("Bilateral symmetry index per joint — reference vs. trained agent", fontsize=14)
    ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3); ax.tick_params(labelsize=10)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=150)
    plt.close(fig)
    print(f"wrote {out.name}.pdf / .png  in {OUT_DIR}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=None, help="reference trajectory .npz (default: trajectory_adapted.npz)")
    ap.add_argument("--agent", default=None,
                    help="PPOJax pkl: roll out the policy (RSI off) and measure ITS symmetry; "
                         "with --npz also present, prints reference-vs-agent and a comparison chart")
    ap.add_argument("--n_steps", type=int, default=400, help="agent rollout length (control steps)")
    args = ap.parse_args()
    _use_times_new_roman()

    # --- reference ---
    npz = Path(args.npz).resolve() if args.npz else paths.TRAJ_ADAPTED
    qpos_ref, freq = load_qpos(npz)
    print(f"\n=== REFERENCE: {npz.name}   frames {qpos_ref.shape[0]} @ {freq:.1f} Hz ===")
    mc_ref, nc_ref = fold_mean_cycle(qpos_ref)
    rows_ref = symmetry_table(mc_ref)
    print_table(rows_ref, nc_ref)

    if not args.agent:
        plot_overlay(mc_ref, rows_ref, OUT_DIR / "gait_symmetry")
        return

    # --- trained agent ---
    print(f"\n=== TRAINED AGENT: {Path(args.agent).parent.name}  (RSI off) ===")
    qpos_ag, dt = agent_qpos(args.agent, n_steps=args.n_steps)
    mc_ag, nc_ag = fold_mean_cycle(qpos_ag)
    rows_ag = symmetry_table(mc_ag)
    print_table(rows_ag, nc_ag)
    plot_overlay(mc_ag, rows_ag, OUT_DIR / "gait_symmetry_agent")
    plot_compare(rows_ref, rows_ag, OUT_DIR / "gait_symmetry_compare")


if __name__ == "__main__":
    main()
