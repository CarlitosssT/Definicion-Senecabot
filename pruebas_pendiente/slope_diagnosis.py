"""
Diagnostico del limite de pendiente: por que se cae la cabra? Compara, en las pendientes limite,
  * agente v140 con friccion x1 / x2 / x4   -> si el limite sube con la friccion, es deslizamiento
  * agente v0218 (0.218 m/s), friccion x1    -> si sube el limite, es la velocidad/dinamica de la marcha
y para cada caso mide:
  * saturacion de actuadores: fraccion de pasos con |accion| >= 0.98 (por cadera / rodilla / tobillo);
    si es alta, el limite es de torque (accion normalizada; ctrlrange = +-80 N.m cadera/rodilla, +-40 tobillo)
  * modo de caida: cabeceo (+ = nariz abajo) y alabeo del tronco respecto a la pendiente, altura del tronco.

    python pruebas_pendiente/slope_diagnosis.py        (PYTHONPATH = raiz del repo y pruebas_pendiente/)
Salida: pruebas_pendiente/slope_diagnosis.json
"""
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from geometry_opt import config as C
from slope_test import tilted_spec

ROOT = Path(__file__).resolve().parent
CASES = [("up", 20), ("up", 25), ("up", 30), ("down", 25), ("down", 30), ("down", 35), ("down", 40)]
VARIANTS = [("v140", 1.0), ("v140", 2.0), ("v140", 4.0), ("v0218", 1.0)]
N_ENVS, STEPS, SAT = 32, 900, 0.98


def euler_from_quat(q):
    """q (...,4) w,x,y,z -> (pitch nariz-abajo, roll) en grados, respecto al marco de la pendiente."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    r20 = 2 * (x * z - w * y)                     # componente z del eje x del cuerpo
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return np.degrees(-np.arcsin(np.clip(r20, -1, 1))), np.degrees(np.arctan2(r21, r22))


def run(agent_conf, agent_state, reference, angle, mode, mu):
    from geometry_opt.simulation import _make_env
    from simulation.analysis.debug_agent import rollout
    env = _make_env(agent_conf, reference, spec=tilted_spec(angle, mode, mu))
    dt = float(env.dt)
    d = rollout(env, agent_conf, agent_state, N_ENVS, STEPS, deterministic=True, seed=0)
    fell = d["absorbing"].astype(bool)
    ever = fell.any(0)
    first = np.where(ever, fell.argmax(0), STEPS)
    alive = np.arange(STEPS)[:, None] < first[None, :]                 # (T, envs)
    a = np.abs(d["action"])                                            # (T, envs, 12): [fr,fl,br,bl] x [hip,knee,ankle]
    sat = (a >= SAT * np.asarray(env.info.action_space.high)).astype(float)
    out = dict(agent=None, angle=angle, mode=mode, mu=mu, fall_frac=float(ever.mean()),
               t_fall_median=float(np.median(first[ever]) * dt) if ever.any() else None)
    for j, name in enumerate(("hip", "knee", "ankle")):
        cols = [3 * k + j for k in range(4)]
        out[f"sat_{name}"] = float((sat[..., cols].mean(-1) * alive).sum() / alive.sum())
    vx = []
    for e in range(N_ENVS):
        n = int(first[e])
        if n > 20:
            dx = np.diff(d["qpos"][:n, e, 0])
            k = np.abs(dx) < 1.0
            vx.append(dx[k].sum() / (k.sum() * dt))
    out["vx_mean"] = float(np.mean(vx)) if vx else None
    if ever.any():
        idx = np.where(ever)[0]
        # estado justo antes de la terminacion (el paso `first` ya es el reinicio)
        q = np.stack([d["qpos"][max(first[e] - 1, 0), e] for e in idx])
        pitch, roll = euler_from_quat(q[:, 3:7])
        out.update(fall_pitch_deg=float(pitch.mean()), fall_abs_roll_deg=float(np.abs(roll).mean()),
                   fall_height_m=float(q[:, 2].mean()))
    return out


def main():
    from loco_mujoco.algorithms import PPOJax
    results = []
    loaded = {}
    for key, mu in VARIANTS:
        if key not in loaded:
            loaded[key] = PPOJax.load_agent(str(C.MODELS[key]["agent"]))
        ac, st = loaded[key]
        for mode, ang in CASES:
            r = run(ac, st, C.MODELS[key]["reference"], ang, mode, mu)
            r["agent"] = key
            print(json.dumps(r), flush=True)
            results.append(r)
            (ROOT / "slope_diagnosis.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
