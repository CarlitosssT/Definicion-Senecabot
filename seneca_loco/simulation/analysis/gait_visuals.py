"""
gait_visuals.py — qualitative gait figures for the trained agent (thesis "Qualitative Gait
Assessment").

Produces two vector PDFs (Times New Roman) in artifacts/figures/thesis_eval/:

  1. gait_filmstrip            : a row of ~6 evenly-spaced rendered poses across ONE gait cycle,
                                 trained agent (top) vs. reference (bottom), same phases.
  2. gait_footfall_diagram     : stance/swing (duty-factor) bars per leg (FR/FL/BR/BL) over the
                                 gait cycle, trained agent vs. reference — shows whether the
                                 policy reproduces the reference footfall sequence & duty factor.

Both are rendered offscreen (EGL). Run from the seneca_loco repo root in the `workspace` env:

    MUJOCO_GL=egl python -m simulation.analysis.gait_visuals \
        --path artifacts/trained_agents/curated/5_best_try/07-58-27/PPOJax_saved.pkl
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")          # offscreen GL on headless boxes

import argparse
from pathlib import Path

import numpy as np
import mujoco
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from scipy.signal import find_peaks

from simulation.config import paths
from simulation.analysis.thesis_eval_figures import build_env
from simulation.analysis.debug_agent import rollout, FOOT_GEOM_NAMES

OUT_DIR = paths.FIGURES / "thesis_eval"
FEET = ["FR", "FL", "BR", "BL"]                     # same order as FOOT_GEOM_NAMES
CONTACT_TOL = 0.012                                 # foot "in stance" when sphere bottom within this of floor
N_FILM = 6                                          # poses per filmstrip row
N_PHASE = 100                                       # gait-cycle resampling resolution


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
# gait-cycle bookkeeping
# ----------------------------------------------------------------------
def _cycle_troughs(fr_hip):
    """Frame indices of FR-hip-angle troughs = gait-cycle boundaries (same convention as the
    augmentation pipeline / joint-tracking plots)."""
    troughs, _ = find_peaks(-np.asarray(fr_hip), prominence=0.04, distance=20)
    return troughs


def _foot_z_from_qpos(model, qpos_seq, foot_ids):
    """Forward-kinematics each frame of a (T, nq) qpos sequence and return (T, 4) foot-geom z."""
    data = mujoco.MjData(model)
    T = qpos_seq.shape[0]
    fz = np.empty((T, len(foot_ids)))
    for f in range(T):
        data.qpos[:] = qpos_seq[f]
        mujoco.mj_forward(model, data)
        fz[f] = data.geom_xpos[foot_ids, 2]
    return fz


def _stance_height_proxy(foot_z, frac=0.30):
    """Per-foot stance proxy for a *kinematic* (non-contact-resolved) clip: a foot is 'planted'
    when its height sits in the lowest ``frac`` of that foot's own vertical range. Needed for the
    reference, whose retargeted feet hover a few cm above the floor and never make physical
    contact, so an absolute floor threshold reads ~0% stance. Captures the footfall *timing/
    sequence* regardless of the clip's vertical offset."""
    lo = np.percentile(foot_z, 2, axis=0)
    hi = np.percentile(foot_z, 98, axis=0)
    thr = lo + frac * np.maximum(hi - lo, 1e-6)
    return foot_z <= thr[None, :]


def _phase_average_stance(stance, troughs):
    """Resample each full gait cycle's stance (T,4)->(N_PHASE,4) and average across cycles.
    Returns (phase_pct(N_PHASE,), duty(N_PHASE,4) in [0,1]) or None if <2 cycles."""
    if len(troughs) < 2:
        return None
    xs = np.linspace(0.0, 1.0, N_PHASE)
    acc = []
    for a, b in zip(troughs[:-1], troughs[1:]):
        seg = stance[a:b].astype(float)                # (L,4)
        if seg.shape[0] < 4:
            continue
        src = np.linspace(0.0, 1.0, seg.shape[0])
        acc.append(np.stack([np.interp(xs, src, seg[:, k]) for k in range(seg.shape[1])], axis=1))
    if not acc:
        return None
    return xs * 100.0, np.mean(acc, axis=0)


# ----------------------------------------------------------------------
# 1. rendered filmstrip
# ----------------------------------------------------------------------
def _render_pose(model, renderer, qpos, az=90.0, elev=-12.0, dist=1.7):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.qpos[:3]                      # track the base so each pose is centred
    cam.lookat[2] = max(float(data.qpos[2]), 0.18)
    cam.distance, cam.azimuth, cam.elevation = dist, az, elev
    renderer.update_scene(data, camera=cam)
    return renderer.render()


def make_filmstrip(model, qpos_env, ref_env, frames, out, az=90.0, dist=1.7):
    """2x N_FILM grid of rendered poses: agent row (top) + reference row (bottom)."""
    renderer = mujoco.Renderer(model, height=480, width=480)
    agent_imgs = [_render_pose(model, renderer, qpos_env[f], az=az, dist=dist) for f in frames]
    ref_imgs = [_render_pose(model, renderer, ref_env[f], az=az, dist=dist) for f in frames]
    renderer.close()

    pct = np.linspace(0, 100, len(frames) + 1)[:-1]    # phase at each shown frame (cycle start=0)
    fig, axes = plt.subplots(2, len(frames), figsize=(2.05 * len(frames), 4.5))
    for j in range(len(frames)):
        axes[0, j].imshow(agent_imgs[j]); axes[1, j].imshow(ref_imgs[j])
        axes[0, j].set_title(f"{pct[j]:.0f}%", fontsize=12)
        for r in (0, 1):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            for sp in axes[r, j].spines.values():
                sp.set_visible(False)
    axes[0, 0].set_ylabel("Trained agent", fontsize=13)
    axes[1, 0].set_ylabel("Reference", fontsize=13)
    fig.suptitle("Rendered gait over one stride — trained agent vs. reference  (% of gait cycle)",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=150)
    plt.close(fig)
    print(f"  wrote {out.name}.pdf / .png")


# ----------------------------------------------------------------------
# 2. footfall / duty-factor diagram
# ----------------------------------------------------------------------
def make_footfall(ag_phase, ag_duty, rf_phase, rf_duty, ag_df, rf_df, out):
    """Stance bars per leg, agent (blue) above reference (orange), across the gait cycle."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    lane_h = 0.34
    yticks, ylabels = [], []
    for i, leg in enumerate(FEET):
        base = i * 1.0                                 # one unit per leg, two lanes inside
        for phase, duty, off, color, tag, df in (
                (ag_phase, ag_duty[:, i], 0.52, "tab:blue", "agent", ag_df[i]),
                (rf_phase, rf_duty[:, i], 0.10, "tab:orange", "ref", rf_df[i])):
            y0 = base + off
            stance = duty > 0.5
            ax.fill_between(phase, y0, y0 + lane_h, where=stance, step="mid",
                            color=color, alpha=0.85, linewidth=0)
            ax.text(101, y0 + lane_h / 2, f"{tag}  D={df*100:.0f}%",
                    va="center", ha="left", fontsize=9, color=color)
            yticks.append(y0 + lane_h / 2); ylabels.append("")
        ax.axhline(base, color="0.85", lw=0.6)
        ax.text(-3, base + 0.5, leg, va="center", ha="right", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 100); ax.set_ylim(-0.1, len(FEET))
    ax.set_yticks([]); ax.set_xlabel("% gait cycle", fontsize=12)
    ax.set_title("Footfall (stance shaded) and duty factor D per leg — agent vs. reference",
                 fontsize=14)
    ax.tick_params(labelsize=11); ax.grid(True, axis="x", alpha=0.3)
    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="tab:blue", alpha=0.85,
                             label="trained agent — simulated ground contact"),
                       Patch(color="tab:orange", alpha=0.85,
                             label="reference — kinematic foot-height proxy")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=150)
    plt.close(fig)
    print(f"  wrote {out.name}.pdf / .png")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=str(paths.TRAINED_AGENTS / "curated" /
                                          "5_best_try/07-58-27/PPOJax_saved.pkl"))
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--n_steps", type=int, default=240)
    ap.add_argument("--az", type=float, default=90.0, help="camera azimuth for the filmstrip")
    ap.add_argument("--dist", type=float, default=1.7, help="camera distance for the filmstrip")
    args = ap.parse_args()
    _use_times_new_roman()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env, ac, as_, cfg = build_env(args.path)
    model = env.get_model()
    foot_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                         for g in FOOT_GEOM_NAMES])
    foot_r = np.array([float(model.geom_size[f][0]) for f in foot_ids])

    print("rolling out (deterministic)...")
    d = rollout(env, ac, as_, args.n_envs, args.n_steps, deterministic=True, seed=0,
                collect_dynamics=True)
    done = d["done"].astype(bool)
    surv = ~done.any(axis=0)
    env_i = int(np.argmax(surv)) if surv.any() else 0
    qpos = np.asarray(d["qpos"])[:, env_i, :]          # (T, nq) agent
    ref = np.asarray(d["ref_qpos"])[:, env_i, :]       # (T, nq) reference

    # --- one gait cycle for the filmstrip (agent FR-hip troughs) ---
    troughs = _cycle_troughs(qpos[:, 7])
    if len(troughs) >= 2:
        s, e = int(troughs[0]), int(troughs[1])
    else:
        s, e = 0, min(64, qpos.shape[0])
    frames = np.linspace(s, e, N_FILM, endpoint=False).astype(int)
    print(f"  representative env {env_i}; cycle frames [{s},{e}) ({e-s} steps); shown {list(frames)}")
    make_filmstrip(model, qpos, ref, frames, OUT_DIR / "gait_filmstrip",
                   az=args.az, dist=args.dist)

    # --- footfall: agent = true simulated ground contact; reference = kinematic foot-height
    #     proxy (the retargeted reference floats a few cm and never physically contacts) ---
    ag_stance = (np.asarray(d["foot_z"])[:, env_i, :] <= (foot_r[None, :] + CONTACT_TOL))
    rf_stance = _stance_height_proxy(_foot_z_from_qpos(model, ref, foot_ids))
    ag = _phase_average_stance(ag_stance, _cycle_troughs(qpos[:, 7]))
    rf = _phase_average_stance(rf_stance, _cycle_troughs(ref[:, 7]))
    if ag is None or rf is None:
        print("  [warn] not enough cycles for footfall diagram"); return
    ag_phase, ag_duty = ag
    rf_phase, rf_duty = rf
    ag_df = ag_stance.mean(axis=0)                     # overall duty factor per leg
    rf_df = rf_stance.mean(axis=0)
    make_footfall(ag_phase, ag_duty, rf_phase, rf_duty, ag_df, rf_df,
                  OUT_DIR / "gait_footfall_diagram")
    print(f"\nFigures in {OUT_DIR}")


if __name__ == "__main__":
    main()
