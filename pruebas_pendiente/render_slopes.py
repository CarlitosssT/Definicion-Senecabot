"""
Videos de las pruebas de pendiente (un robot, arranque en la fase 0 de la referencia).

La simulacion corre en el marco de la pendiente (gravedad inclinada, ver slope_test.py); para el
video se gira todo el mundo (robot + suelo) alrededor del origen para que la gravedad quede vertical,
asi el suelo se ve inclinado. El video se corta 0.5 s despues de la primera caida.

    MUJOCO_GL=egl python pruebas_pendiente/render_slopes.py [--agent v140|<ruta .pkl>] [--cases up:5 ... down:20]
Salida: pruebas_pendiente/videos/<up|down>_<grados>.mp4 (v140) o videos/<etiqueta>/<up|down>_<grados>.mp4
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import imageio_ffmpeg
import mediapy
import mujoco
import numpy as np

from geometry_opt import config as C
from slope_test import load_agent, make_env

mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())      # no hace falta ffmpeg en el sistema
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "videos"


def world_rotation(angle_deg, mode):
    """Giro alrededor de y que lleva la gravedad simulada a -z. Ascenso: alpha=-theta."""
    a = np.deg2rad(angle_deg)
    alpha = -a if mode == "up" else a
    return alpha, np.array([np.cos(alpha / 2), 0.0, np.sin(alpha / 2), 0.0])   # quat (w,x,y,z)


def rotate_qpos(q, alpha, qw):
    c, s = np.cos(alpha), np.sin(alpha)
    out = q.copy()
    x, z = q[:, 0], q[:, 2]
    out[:, 0], out[:, 2] = x * c + z * s, -x * s + z * c
    res = np.zeros((len(q), 4))
    for i in range(len(q)):
        mujoco.mju_mulQuat(res[i], qw, q[i, 3:7])
    out[:, 3:7] = res
    return out


def render(qpos, angle, mode, fps, out):
    alpha, qw = world_rotation(angle, mode)
    spec = mujoco.MjSpec.from_file(str(C.XML_PATH))
    spec.option.gravity = [0, 0, -9.81]
    floor = next(g for g in spec.geoms if g.name == "floor")
    floor.quat = qw
    model = spec.compile()
    q = rotate_qpos(qpos, alpha, qw)
    data = mujoco.MjData(model)
    H, W = 480, 854
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90.0, -10.0, 2.2
    frames = []
    for row in q:
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    renderer.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(out, frames, fps=fps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="v140", help="v140, v0218 o ruta a un PPOJax_saved.pkl")
    p.add_argument("--cases", nargs="+",
                   default=[f"{m}:{a}" for m in ("up", "down") for a in (5, 10, 15, 20)] + ["up:25", "down:30"])
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=50)
    a = p.parse_args()

    from simulation.analysis.debug_agent import rollout
    agent_conf, agent_state, label, reference = load_agent(a.agent)
    out_dir = OUT if label == "v140" else OUT / label
    for case in a.cases:
        mode, ang = case.split(":")
        ang = float(ang)
        env = make_env(agent_conf, reference, ang, mode)
        env.th.random_start = False
        hz = 1.0 / float(env.dt)
        n = int(a.seconds * hz)
        d = rollout(env, agent_conf, agent_state, 4, n, deterministic=True, seed=0)
        fell = d["absorbing"].astype(bool)[:, 0]
        end = int(fell.argmax()) + int(0.5 * hz) if fell.any() else n
        end = min(end, n)
        step = max(int(round(hz / a.fps)), 1)
        out = out_dir / f"{mode}_{int(ang)}.mp4"
        render(d["qpos"][:end:step, 0, :], ang, mode, a.fps, out)
        print(f"{out.name}: {'cae en %.2f s' % (fell.argmax() / hz) if fell.any() else 'no cae'}", flush=True)


if __name__ == "__main__":
    main()
