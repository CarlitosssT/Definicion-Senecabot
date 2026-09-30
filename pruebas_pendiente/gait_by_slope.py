"""
Marcha de un agente (por defecto v140, sin ajuste fino) en cada inclinacion: ciclo de marcha promedio y
metricas por ciclo.

Ciclo de marcha = entre dos aterrizajes consecutivos de la pata delantera derecha (FR). Contacto de una
pata: su esfera toca el suelo (centro a <= 2.7 cm) y sigue en contacto hasta subir de 3.7 cm (histeresis
contra el parpadeo). Se descartan los primeros 1.0 s (transitorio del reinicio) y todo lo posterior a una
caida (criterio de caida del propio agente). Pendiente con signo: + sube, - baja.

Por pendiente guarda:
  * ciclo promedio (101 puntos, 0-100 % del ciclo) de los 12 angulos articulares y de la probabilidad de
    apoyo de cada pata (media y desviacion estandar entre ciclos);
  * por ciclo: periodo, longitud de zancada, velocidad, factor de apoyo de cada pata, cabeceo del tronco
    respecto al suelo y respecto a la horizontal, altura del tronco, potencia mecanica, costo de transporte
    y velocidad de resbalamiento de cada pata apoyada.

    python pruebas_pendiente/gait_by_slope.py [--agent v140] [--angles -20 -15 ... 20]
Salida: pruebas_pendiente/gait_by_slope.npz (v140) o gait_by_slope_<etiqueta>.npz
"""
import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from geometry_opt import config as C                     # noqa: E402
from slope_test import load_agent, make_env             # noqa: E402

ROOT = Path(__file__).resolve().parent
FEET = ("fr_foot", "fl_foot", "br_foot", "bl_foot")
JOINTS = [f"{leg}_{j}" for leg in ("fr", "fl", "br", "bl") for j in ("hip", "knee", "ankle")]
TOUCH, RELEASE = 0.027, 0.037          # [m] altura del centro de la esfera del pie (radio 0.025)
N_PHASE = 101


def contacts(foot_z):
    """foot_z (T, 4) -> contacto con histeresis (T, 4)."""
    c = np.zeros(foot_z.shape, dtype=bool)
    state = foot_z[0] <= RELEASE
    for t in range(len(foot_z)):
        state = np.where(state, foot_z[t] <= RELEASE, foot_z[t] <= TOUCH)
        c[t] = state
    return c


def foot_kinematics(model, qpos, qvel):
    """Altura y velocidad (marco del mundo = pendiente) de los 4 pies, por cinematica en CPU."""
    d = mujoco.MjData(model)
    ids = [model.site(n).id for n in FEET]
    z, v, v6 = np.zeros((len(qpos), 4)), np.zeros((len(qpos), 4, 3)), np.zeros(6)
    for t in range(len(qpos)):
        d.qpos[:], d.qvel[:] = qpos[t], qvel[t]
        mujoco.mj_fwdPosition(model, d)
        mujoco.mj_fwdVelocity(model, d)
        for i, s in enumerate(ids):
            z[t, i] = d.site_xpos[s, 2]
            mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_SITE, s, v6, 0)
            v[t, i] = v6[3:]
    return z, v


def resample(x, n=N_PHASE):
    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, x[:, k].astype(float)) for k in range(x.shape[1])], axis=1)


def analyse(slope, qpos, qvel, tau, alive, model, mass, dt, warmup):
    """Ciclos de todas las envs de una pendiente."""
    joint_qpos = model.jnt_qposadr[[model.joint(f"{j}_joint").id for j in JOINTS]]
    joint_qvel = model.jnt_dofadr[[model.joint(f"{j}_joint").id for j in JOINTS]]
    cyc_q, cyc_c, rows = [], [], []
    for e in range(qpos.shape[1]):
        n = int(alive[:, e].sum())
        if n <= warmup + 10:
            continue
        q, qd, tq = qpos[warmup:n, e], qvel[warmup:n, e], tau[warmup:n, e]
        fz, fv = foot_kinematics(model, q, qd)
        c = contacts(fz)
        td = np.where(c[1:, 0] & ~c[:-1, 0])[0] + 1              # aterrizajes de FR
        eul = R.from_quat(q[:, [4, 5, 6, 3]]).as_euler("xyz", degrees=True)
        nose_up = -eul[:, 1]                                      # cabeceo respecto al suelo (+ nariz arriba)
        power = np.abs(tq * qd[:, joint_qvel]).sum(1)
        slip = np.linalg.norm(fv[..., :2], axis=-1)
        touching = fz <= TOUCH
        for a, b in zip(td[:-1], td[1:]):
            if b - a < 10:
                continue
            if np.abs(np.diff(q[a:b + 1, 0])).max() > 0.1:       # salto de vuelta de la referencia (>10 m/s)
                continue
            period = (b - a) * dt
            stride = q[b, 0] - q[a, 0]
            speed = stride / period
            p = power[a:b].mean()
            cyc_q.append(resample(np.rad2deg(q[a:b + 1, joint_qpos])))
            cyc_c.append(resample(c[a:b + 1]))
            rows.append([period, stride, speed, *c[a:b].mean(0), nose_up[a:b].mean(), nose_up[a:b].mean() + slope,
                         q[a:b, 2].mean(), p, p / (mass * 9.81 * max(speed, 1e-3)),
                         *[np.median(slip[a:b, i][touching[a:b, i]]) if touching[a:b, i].any() else np.nan
                           for i in range(4)]])
    return np.array(cyc_q), np.array(cyc_c), np.array(rows)


METRICS = ["period_s", "stride_m", "speed_mps", "duty_fr", "duty_fl", "duty_br", "duty_bl",
           "pitch_ground_deg", "pitch_horizontal_deg", "height_m", "power_W", "cot",
           "slip_fr_mps", "slip_fl_mps", "slip_br_mps", "slip_bl_mps"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="v140", help="v140, v0218 o ruta a un PPOJax_saved.pkl")
    p.add_argument("--angles", type=float, nargs="+", default=[-20, -15, -10, -5, 0, 5, 10, 15, 20])
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--warmup", type=float, default=1.0, help="segundos iniciales descartados")
    a = p.parse_args()

    from simulation.analysis.debug_agent import rollout
    agent_conf, agent_state, label, reference = load_agent(a.agent)
    model = mujoco.MjModel.from_xml_path(str(C.XML_PATH))
    mass = float(model.body_subtreemass[model.body("base").id])
    out = dict(angles=np.array(a.angles), joints=np.array(JOINTS), feet=np.array(FEET),
               metrics=np.array(METRICS), agent=label)
    for s in a.angles:
        env = make_env(agent_conf, reference, abs(s), "up" if s >= 0 else "down")
        dt = float(env.dt)
        steps = int(a.seconds / dt)
        d = rollout(env, agent_conf, agent_state, a.n_envs, steps, deterministic=True, seed=0,
                    collect_dynamics=True)
        fell = d["absorbing"].astype(bool)
        first = np.where(fell.any(0), fell.argmax(0), steps)
        alive = np.arange(steps)[:, None] < first[None, :]
        nu = len(JOINTS)
        act_dof = [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(nu)]
        tau = d["qfrc_actuator"][..., act_dof]
        cq, cc, rows = analyse(s, d["qpos"], d["qvel"], tau, alive, model, mass, dt, int(a.warmup / dt))
        key = f"s{int(s):+d}"
        out[f"{key}_joint_mean"], out[f"{key}_joint_std"] = cq.mean(0), cq.std(0)
        out[f"{key}_contact_mean"] = cc.mean(0)
        out[f"{key}_metrics"] = rows
        out[f"{key}_fall_frac"] = float(fell.any(0).mean())
        # un tramo de contactos de la env 0 (patron de pisadas), si sobrevivio
        e0 = int(np.argmax(first))
        q0 = d["qpos"][int(a.warmup / dt):first[e0], e0][:200]
        out[f"{key}_footfalls"] = contacts(foot_kinematics(model, q0, d["qvel"][int(a.warmup / dt):first[e0], e0][:200])[0])
        print(f"pendiente {s:+5.1f}: {len(rows)} ciclos, caidas {out[f'{key}_fall_frac']:.0%}, "
              f"periodo {np.median(rows[:, 0]):.3f} s, zancada {np.median(rows[:, 1]):.3f} m, "
              f"v {np.median(rows[:, 2]):.2f} m/s, apoyo {np.round(np.median(rows[:, 3:7], 0), 2)}", flush=True)
    name = "gait_by_slope.npz" if label == "v140" else f"gait_by_slope_{label}.npz"
    np.savez_compressed(ROOT / name, **out)
    print(f"-> {ROOT / name}")


if __name__ == "__main__":
    main()
