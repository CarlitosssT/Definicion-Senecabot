"""
gait_targets_figure.py — "imitation targets" schematic (thesis Idea A).

Renders the SenecaBot in a representative gait pose and marks, on the body, the three kinds of
quantity the imitation reward tracks, so the reader can see at a glance *what the agent imitates*:

  * the 13 tracked body SITES (base + {fr,fl,br,bl}x{foot,shank,thigh}) as colour-coded spheres
    -> site position / orientation / velocity targets  (reward: rpos / rquat / rvel),
  * the 12 hinge JOINTS, shown via MuJoCo joint visualisation -> joint-angle targets (reward: qpos),
  * the base site -> root-pose target (reward: root tracking).

Output: a Times-New-Roman vector PDF/PNG. Run from the seneca_loco repo root:
    MUJOCO_GL=egl python -m simulation.analysis.gait_targets_figure [--az 140 --elev -18 --dist 1.6]
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import numpy as np
import mujoco
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

from simulation.config import paths

XML = paths.XML_PATH
OUT_DIR = paths.FIGURES / "thesis_eval"

GROUP_COLOR = {                                   # site group -> RGBA
    "foot":  (0.85, 0.15, 0.15, 1.0),             # red
    "shank": (0.95, 0.55, 0.10, 1.0),             # orange
    "thigh": (0.15, 0.35, 0.85, 1.0),             # blue
    "base":  (0.10, 0.65, 0.20, 1.0),             # green
}


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


def _site_group(name):
    for g in GROUP_COLOR:
        if g in name:
            return g
    return None


def world_to_pixel(pts, cam, fovy_deg, W, H):
    """Project Nx3 world points to pixel coords for a MuJoCo free camera. Returns (N,3) = (px,py,depth)."""
    az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # cam -> scene
    cam_pos = np.array(cam.lookat) - cam.distance * forward
    right = np.cross(forward, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    f = 0.5 * H / np.tan(np.radians(fovy_deg) / 2)
    out = []
    for p in np.atleast_2d(pts):
        d = np.asarray(p, float) - cam_pos
        zc = d @ forward
        out.append((W / 2 + f * (d @ right) / zc, H / 2 - f * (d @ up) / zc, zc))
    return np.array(out)


def representative_qpos():
    """A clear mid-stride pose: ~30% into one gait cycle of the reference trajectory."""
    from loco_mujoco.trajectory import Trajectory
    q = np.asarray(Trajectory.load(str(paths.TRAJ_ADAPTED)).data.qpos)
    troughs, _ = find_peaks(-q[:, 7], prominence=0.04, distance=20)
    if len(troughs) >= 2:
        s, e = int(troughs[0]), int(troughs[1])
        f = s + int(0.30 * (e - s))
    else:
        f = 30
    return q[f]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--az", type=float, default=140.0)
    ap.add_argument("--elev", type=float, default=-18.0)
    ap.add_argument("--dist", type=float, default=1.55)
    ap.add_argument("--site_size", type=float, default=0.020)
    args = ap.parse_args()
    _use_times_new_roman()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qpos[:] = representative_qpos()
    mujoco.mj_forward(model, data)
    W, H = 1000, 760
    JCOL = (0.65, 0.0, 0.65)

    # render a clean, lightly-transparent robot; sites/joints are overlaid in matplotlib (no occlusion)
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[gid][3] = 0.55
    for sid in range(model.nsite):                       # hide all native sites; we draw our own
        model.site_rgba[sid] = [0, 0, 0, 0]

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.qpos[:3]
    cam.lookat[2] = max(float(data.qpos[2]) - 0.12, 0.10)
    cam.distance, cam.azimuth, cam.elevation = args.dist, args.az, args.elev
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    renderer.close()

    # gather target positions in world coords
    site_pts = {g: [] for g in GROUP_COLOR}
    for sid in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        g = _site_group(name) if (name and name.endswith("_mimic")) else None
        if g is not None:
            site_pts[g].append(data.site_xpos[sid])
    hinge = {}                                            # hip/knee/ankle anchors, per leg
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtGeom.mjGEOM_PLANE:  # placeholder, replaced below
            pass
    hinge_pts, fr_leg = [], {}
    for jid in range(model.njnt):
        if model.jnt_type[jid] != 3:                     # 3 = hinge
            continue
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        hinge_pts.append(data.xanchor[jid])
        if jn.startswith("fr_"):
            fr_leg[jn.split("_")[1]] = data.xanchor[jid]   # hip/knee/ankle of the front-right leg

    fig, ax = plt.subplots(figsize=(10, 7.4))
    ax.imshow(img); ax.set_axis_off()
    ax.set_title("Imitation targets served by the reference trajectory", fontsize=16, pad=10)

    allpx = []
    # overlay hinge joints first (magenta diamonds), then sites on top so colours aren't hidden
    hp = world_to_pixel(np.array(hinge_pts), cam, 45.0, W, H)
    ax.scatter(hp[:, 0], hp[:, 1], s=70, marker="D", c=[JCOL], edgecolors="white",
               linewidths=0.8, zorder=4)
    allpx.append(hp[:, :2])
    for g, pts in site_pts.items():
        if not pts:
            continue
        px = world_to_pixel(np.array(pts), cam, 45.0, W, H)
        allpx.append(px[:, :2])
        s = 260 if g == "base" else 155
        mk = "*" if g == "base" else "o"
        ax.scatter(px[:, 0], px[:, 1], s=s, marker=mk, c=[GROUP_COLOR[g][:3]],
                   edgecolors="white", linewidths=1.3, zorder=6)
    # label hip/knee/ankle on the front-right leg
    for jn, p in fr_leg.items():
        q = world_to_pixel(np.array([p]), cam, 45.0, W, H)[0]
        ax.annotate(jn, (q[0], q[1]), xytext=(13, -2), textcoords="offset points",
                    fontsize=10.5, color="white", va="center", fontweight="bold", zorder=8,
                    path_effects=[__import__("matplotlib.patheffects", fromlist=["withStroke"])
                                  .withStroke(linewidth=2.4, foreground=JCOL)])
    # auto-crop to the robot (+ padding), clipped to the image
    allpx = np.vstack(allpx)
    pad = 90
    x0, x1 = max(allpx[:, 0].min() - pad, 0), min(allpx[:, 0].max() + pad, W)
    y0, y1 = max(allpx[:, 1].min() - pad, 0), min(allpx[:, 1].max() + pad * 1.4, H)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)

    site_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GROUP_COLOR["foot"][:3],
               markeredgecolor="white", markersize=11, label="foot sites (4)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GROUP_COLOR["shank"][:3],
               markeredgecolor="white", markersize=11, label="shank sites (4)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GROUP_COLOR["thigh"][:3],
               markeredgecolor="white", markersize=11, label="thigh sites (4)"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=GROUP_COLOR["base"][:3],
               markeredgecolor="white", markersize=16, label="base site (root pose)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=JCOL,
               markeredgecolor="white", markersize=9, label="hinge joints (12 angles)"),
    ]
    leg1 = ax.legend(handles=site_handles, title="Tracked quantities",
                     loc="upper left", fontsize=11, title_fontsize=12, frameon=True,
                     framealpha=0.92, borderpad=0.8)
    ax.add_artist(leg1)

    note = ("Reward terms (subsec. 2.4.6):\n"
            r"$\bullet$ joint angles $\rightarrow$ qpos" "\n"
            r"$\bullet$ root pose $\rightarrow$ root tracking" "\n"
            r"$\bullet$ site position $\rightarrow$ rpos" "\n"
            r"$\bullet$ site orientation $\rightarrow$ rquat" "\n"
            r"$\bullet$ site velocity $\rightarrow$ rvel")
    ax.text(0.985, 0.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))

    fig.tight_layout()
    out = OUT_DIR / "schematic__imitation_targets__v1"
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)
    print(f"wrote {out.name}.pdf / .png in {OUT_DIR}")


if __name__ == "__main__":
    main()
