"""
Prueba de pendientes: ascenso y descenso a pendientes fijas, fraccion de caidas y velocidad.

La pendiente se aplica inclinando el vector de gravedad en un mundo con suelo plano (equivalente
fisico a inclinar el plano: el angulo gravedad-suelo es el mismo). Asi el mundo queda en el marco
de la pendiente: la altura z observada por la politica es la altura sobre el suelo y no crece al
subir, y el manejador de terminacion por defecto (pose de la raiz vs. referencia) sigue siendo valido.

  * v140 / v0218 (sin SlopeRandomizer): la gravedad inclinada va en el modelo; la politica NO percibe
    la pendiente (nunca fue entrenada con ella).
  * agentes del ajuste fino (conf_finetune_slope.yaml, con SlopeRandomizer + gravity_obs): la pendiente
    se fija con el propio randomizador (rango [s, s]), asi la politica la percibe por su observacion.
    --via-randomizer fuerza este camino tambien para v140 (misma fisica; sirve de verificacion).

Criterio de caida (--fall): "agent" (por defecto) usa el del propio agente: v140 = tronco a mas de 30 grados
de la orientacion de la referencia (paralelo a la pendiente); ajuste fino con recompensa de tarea = tronco
a mas de 50 grados de la vertical (gravedad) o raiz a menos de 0.15 m. Para comparar agentes entre si,
usar el mismo en todos: --fall upright o --fall reference.

    python pruebas_pendiente/slope_test.py [--agent v140|v0218|<ruta .pkl>] [--angles 0 5 10 15 20]
Salida: slope_results.json (v140) o slope_results_<etiqueta>.json
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


def tilted_spec(angle_deg, mode, mu_scale=1.0):
    """mode 'up': el robot (avanza hacia +x) sube; 'down': baja.
    mu_scale multiplica la friccion de deslizamiento del suelo y de los pies (diagnostico)."""
    spec = mujoco.MjSpec.from_file(str(C.XML_PATH))
    if mu_scale != 1.0:
        for g in spec.geoms:
            if g.name in ("floor", "fr_foot", "fl_foot", "br_foot", "bl_foot"):
                f = np.array(g.friction, dtype=float)
                f[0] *= mu_scale
                g.friction = f
    a = np.deg2rad(angle_deg)
    sx = -np.sin(a) if mode == "up" else np.sin(a)
    spec.option.gravity = 9.81 * np.array([sx, 0.0, -np.cos(a)])
    return spec


def load_agent(agent):
    """agent: clave de geometry_opt (v140, v0218) o ruta a un .pkl. Devuelve (conf, estado, etiqueta, referencia)."""
    from loco_mujoco.algorithms import PPOJax
    if agent in C.MODELS:
        path, label, reference = C.MODELS[agent]["agent"], agent, C.MODELS[agent]["reference"]
    else:
        path = Path(agent).resolve()
        label = f"{path.parent.parent.name}_{path.parent.name}"      # fecha_hora del entrenamiento
        if path.stem != "PPOJax_saved":                               # p. ej. PPOJax_final -> ..._final
            label += "_" + path.stem.replace("PPOJax_", "")
        reference = C.MODELS["v140"]["reference"]                     # el ajuste fino usa la de 1.40 m/s
    conf, state = PPOJax.load_agent(str(path))
    return conf, state, label, reference


def uses_slope_randomizer(agent_conf):
    return agent_conf.config.experiment.env_params.get("domain_randomization_type") == "SlopeRandomizer"


FALL_CRITERIA = {
    "upright": dict(terminal_state_type="UprightTerminalStateHandler",
                    terminal_state_params=dict(max_tilt_deg=50.0, min_height=0.15)),
    "reference": dict(terminal_state_type="RootPoseTrajTerminalStateHandler", terminal_state_params=None),
}


def make_env(agent_conf, reference, angle, mode, mu_scale=1.0, via_randomizer=None, fall="agent"):
    """Entorno del agente sobre una pendiente fija (ver docstring del modulo)."""
    from omegaconf import OmegaConf
    from loco_mujoco import TaskFactory
    from loco_mujoco.trajectory.dataclasses import Trajectory
    from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf

    cfg = OmegaConf.to_container(agent_conf.config.experiment.env_params, resolve=True)
    cfg["headless"] = True
    if fall != "agent":
        cfg.update(FALL_CRITERIA[fall])
    if via_randomizer is None:
        via_randomizer = uses_slope_randomizer(agent_conf)
    if via_randomizer:
        s = angle if mode == "up" else -angle                          # + sube, - baja
        cfg.update(domain_randomization_type="SlopeRandomizer",
                   domain_randomization_params=dict(slope_range_deg=[s, s]))
        cfg["spec"] = tilted_spec(0.0, mode, mu_scale)
    else:
        cfg["spec"] = tilted_spec(angle, mode, mu_scale)
    return TaskFactory.get_factory_cls("ImitationFactory").make(
        **cfg, custom_dataset_conf=CustomDatasetConf(traj=Trajectory.load(str(reference))))


def run_case(agent_conf, agent_state, reference, angle, mode, n_envs, steps, seed, via_randomizer=None,
             fall="agent"):
    from simulation.analysis.debug_agent import rollout
    env = make_env(agent_conf, reference, angle, mode, via_randomizer=via_randomizer, fall=fall)
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
    p.add_argument("--agent", default="v140", help="v140, v0218 o ruta a un PPOJax_saved.pkl")
    p.add_argument("--angles", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fall", choices=["agent", "upright", "reference"], default="agent",
                   help="criterio de caida (ver docstring); el mismo para todos los agentes al compararlos")
    p.add_argument("--via-randomizer", action="store_true",
                   help="fija la pendiente con SlopeRandomizer tambien para agentes sin el (verificacion)")
    a = p.parse_args()

    agent_conf, agent_state, label, reference = load_agent(a.agent)
    via = True if a.via_randomizer else None
    results = []
    for mode in ("up", "down"):
        for ang in a.angles:
            if ang == 0 and mode == "down":
                continue
            r = run_case(agent_conf, agent_state, reference, ang, mode, a.n_envs, a.steps, a.seed, via, a.fall)
            r["agent"], r["fall_criterion"] = label, a.fall
            print(json.dumps(r), flush=True)
            results.append(r)
    suffix = ("_via_randomizer" if a.via_randomizer else "") + ("" if a.fall == "agent" else f"_fall-{a.fall}")
    name = "slope_results.json" if label == "v140" and not suffix else f"slope_results_{label}{suffix}.json"
    (ROOT / name).write_text(json.dumps(results, indent=1))
    print(f"-> {ROOT / name}")


if __name__ == "__main__":
    main()
