"""
render_clips.py — two side-by-side-comparable 10 s MuJoCo clips:

  * reference_clip.mp4 : the reference trajectory played back from step 0,
  * policy_clip.mp4    : the trained policy evaluated with RSI OFF (so it resets to the SAME
                         canonical step-0 pose),

both rendered with an identical base-following side camera so the ideal motion and the learned
motion start from the same pose and can be compared frame-for-frame.

Run from the seneca_loco repo root:
    MUJOCO_GL=egl python -m simulation.analysis.render_clips \
        --agent artifacts/trained_agents/curated/5_best_try/07-58-27/PPOJax_saved.pkl
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import numpy as np
import mujoco
import mediapy

from simulation.config import paths

OUT_DIR = paths.RECORDINGS


def render_clip(model, qpos_seq, out, fps, az, elev, dist, lookat_z, H=480, W=854):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = az, elev, dist
    frames = []
    for q in qpos_seq:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[:3]          # follow the base in x,y
        cam.lookat[2] = lookat_z               # fixed height so the robot is framed consistently
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    renderer.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(out, frames, fps=fps)
    print(f"  wrote {out}  ({len(frames)} frames @ {fps} fps)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", default=str(paths.TRAINED_AGENTS / "curated" /
                                           "5_best_try/07-58-27/PPOJax_saved.pkl"))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--az", type=float, default=90.0)
    ap.add_argument("--elev", type=float, default=-12.0)
    ap.add_argument("--dist", type=float, default=1.8)
    ap.add_argument("--lookat_z", type=float, default=0.22)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(paths.XML_PATH))
    camkw = dict(fps=args.fps, az=args.az, elev=args.elev, dist=args.dist, lookat_z=args.lookat_z)

    # ---- reference clip: trajectory_adapted from step 0 ----
    from loco_mujoco.trajectory import Trajectory
    traj = Trajectory.load(str(paths.TRAJ_ADAPTED))
    ref_q = np.asarray(traj.data.qpos)
    ref_hz = float(traj.info.frequency)
    n_ref = int(args.seconds * ref_hz)
    step_ref = max(int(round(ref_hz / args.fps)), 1)        # decimate to target fps
    print("reference clip:")
    render_clip(model, ref_q[0:n_ref:step_ref], OUT_DIR / "reference_clip.mp4", **camkw)

    # ---- policy clip: RSI off so it starts at the same step-0 pose ----
    from simulation.analysis.thesis_eval_figures import build_env
    from simulation.analysis.debug_agent import rollout
    env, ac, as_, cfg = build_env(args.agent)
    env.th.random_start = False                              # turn OFF RSI
    ctrl_hz = 1.0 / float(env.dt)
    n_pol = int(args.seconds * ctrl_hz)
    d = rollout(env, ac, as_, 4, n_pol, deterministic=True, seed=0)
    done = np.asarray(d["done"]).astype(bool)[:, 0]
    end = int(np.argmax(done)) if done.any() else n_pol      # stop at first fall (avoid a reset jump)
    if end < 5:
        end = n_pol
    pol_q = np.asarray(d["qpos"])[:end, 0, :]
    step_pol = max(int(round(ctrl_hz / args.fps)), 1)
    print(f"policy clip (survived {end}/{n_pol} steps = {end*float(env.dt):.1f} s):")
    render_clip(model, pol_q[::step_pol], OUT_DIR / "policy_clip.mp4", **camkw)

    print(f"\nClips in {OUT_DIR}")


if __name__ == "__main__":
    main()
