"""
Prueba de pendientes del agente v140 (100M @ 1.40 m/s): ascenso y descenso a 5/10/15/20 grados.

La pendiente se aplica inclinando el vector de gravedad en un mundo con suelo plano (equivalente
fisico a inclinar el plano: el angulo gravedad-suelo es el mismo). Asi el mundo queda en el marco
de la pendiente: la altura z observada por la politica es la altura sobre el suelo y no crece al
subir, y el manejador de terminacion por defecto (pose de la raiz vs. referencia) sigue siendo valido.
Consecuencia: la politica NO percibe la pendiente (nunca fue entrenada con ella).

    python pruebas_pendiente/slope_test.py [--angles 0 5 10 15 20] [--n-envs 32] [--steps 900]
"""
import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parent
from geometry_opt import config as C          # noqa: E402


def tilted_spec(angle_deg, mode):
    """mode 'up': el robot (avanza hacia +x) sube; 'down': baja."""
    spec = mujoco.MjSpec.from_file(str(C.XML_PATH))
    a = np.deg2rad(angle_deg)
    sx = -np.sin(a) if mode == "up" else np.sin(a)
    spec.option.gravity = 9.81 * np.array([sx, 0.0, -np.cos(a)])
    return spec


def run_case(agent_conf, agent_state, reference, angle, mode, n_envs, steps, seed):
    from geometry_opt.simulation import _make_env
    from simulation.analysis.debug_agent import rollout
    cfg = agent_conf.config.experiment.env_params
    env = _make_env(agent_conf, reference, spec=tilted_spec(angle, mode))
    dt = float(env.dt)
    d = rollout(env, agent_conf, agent_state, n_envs, steps, deterministic=True, seed=seed)
    fell = d["absorbing"].astype(bool)                       # terminacion por caida (no por horizonte)
    ever = fell.any(0)
    first = np.where(ever, fell.argmax(0), steps)            # paso de la primera caida
    # velocidad media en x (marco de la pendiente) mientras no cae
    vx = []
    for e in range(n_envs):
        n = int(first[e])
        if n > 20:
            x = d["qpos"][:n, e, 0]
            keep = np.abs(np.diff(x)) < 1.0                  # descarta saltos de vuelta de la referencia
            vx.append(np.diff(x)[keep].sum() / (keep.sum() * dt))
    return dict(angle=angle, mode=mode, n_envs=n_envs,
                fall_frac=float(ever.mean()),
                t_fall_median=float(np.median(first[ever]) * dt) if ever.any() else None,
                t_fall_min=float(first[ever].min() * dt) if ever.any() else None,
                survive_s=float(np.mean(first) * dt), t_total=steps * dt,
                vx_mean=float(np.mean(vx)) if vx else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--angles", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    from loco_mujoco.algorithms import PPOJax
    spec = C.MODELS["v140"]
    agent_conf, agent_state = PPOJax.load_agent(str(spec["agent"]))
    results = []
    for mode in ("up", "down"):
        for ang in a.angles:
            if ang == 0 and mode == "down":
                continue
            r = run_case(agent_conf, agent_state, spec["reference"], ang, mode, a.n_envs, a.steps, a.seed)
            print(json.dumps(r), flush=True)
            results.append(r)
    (ROOT / "slope_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
