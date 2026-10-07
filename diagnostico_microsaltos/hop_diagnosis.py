"""
Diagnostico de microsaltos: compara las pisadas de un agente (por defecto v140, terreno plano) con las de su
trayectoria de referencia, pata por pata.

Contacto de una pata: misma definicion que pruebas_pendiente/gait_by_slope.py (centro de la esfera del pie a
<= 2.7 cm del suelo, y sigue en contacto hasta subir de 3.7 cm). Un "vuelo" es un tramo sin contacto entre
dos apoyos; un microsalto es un vuelo corto (< --short s).

Por pata, para el agente (determinista y estocastico) y para la referencia (trayectoria completa; en ella el
apoyo es la etiqueta relativa de MimicReward, porque sus pies delanteros no bajan hasta el suelo):
  * duracion de los vuelos (mediana, % < 0.1 s, % < --short s) y altura maxima del pie en los vuelos cortos;
  * aterrizajes por ciclo de marcha (ciclo = periodo de la referencia; 1.0 = una pisada por ciclo);
  * velocidad vertical del pie al aterrizar;
  * etiqueta de apoyo que usa MimicReward (contact_ref_frac) frente a la de altura con histeresis, y
    coincidencia del agente con la etiqueta de MimicReward (lo que premia contact_w_sum).
Para el agente, ademas: rms de la velocidad vertical del tronco y % de pasos con cada accion en su limite.

    python diagnostico_microsaltos/hop_diagnosis.py [--agent v140] [--n-envs 32] [--seconds 10]
Salida: diagnostico_microsaltos/hop_diagnosis_<etiqueta>.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent), str(ROOT.parent / "pruebas_pendiente")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from geometry_opt import config as C                              # noqa: E402
from slope_test import load_agent, make_env                      # noqa: E402
from gait_by_slope import FEET, JOINTS, contacts, foot_kinematics  # noqa: E402

LEGS = ("fr", "fl", "br", "bl")


def runs(mask):
    """Tramos de True de un vector booleano -> lista de (inicio, fin) con fin exclusivo."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def foot_stats(z, vz, c, dt, short, cycle_s, valid=None):
    """z, vz, c: (T,) de una pata. valid: (T,) pasos utilizables (None = todos). Vuelos que tocan el borde
    de un tramo valido o un paso no valido se descartan (no se sabe su duracion)."""
    valid = np.ones(len(z), bool) if valid is None else valid
    flights, peak, td_vz = [], [], []
    for a, b in runs(~c):
        if a == 0 or b == len(c) or not valid[a - 1:b + 1].all():
            continue
        flights.append((b - a) * dt)
        peak.append(z[a:b].max())
        td_vz.append(vz[b])
    flights, peak = np.array(flights), np.array(peak)
    sh = flights < short
    t_valid = valid.sum() * dt
    return dict(n_flights=int(len(flights)),
                flight_median_s=float(np.median(flights)) if len(flights) else None,
                pct_flights_lt_0p1s=float(100 * (flights < 0.1).mean()) if len(flights) else None,
                pct_flights_lt_short=float(100 * sh.mean()) if len(flights) else None,
                short_flight_peak_cm=float(100 * np.median(peak[sh])) if sh.any() else None,
                touchdowns_per_cycle=float(len(flights) / (t_valid / cycle_s)) if t_valid > 0 else None,
                touchdown_vz_median_mps=float(np.median(td_vz)) if td_vz else None,
                stance_frac=float(c[valid].mean()))


def reference_labels(ref_z, frac):
    """Etiqueta de apoyo de MimicReward: pie en el contact_ref_frac mas bajo de su rango de altura."""
    lo, hi = ref_z.min(0), ref_z.max(0)
    return ref_z <= lo + frac * (hi - lo)


def reference_cycle_s(ref_c, dt):
    """Periodo de la referencia = mediana entre aterrizajes consecutivos de FR."""
    td = np.where(ref_c[1:, 0] & ~ref_c[:-1, 0])[0] + 1
    return float(np.median(np.diff(td)) * dt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="v140", help="v140, v0218 o ruta a un PPOJax_saved.pkl")
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--warmup", type=float, default=1.0, help="segundos iniciales descartados")
    p.add_argument("--short", type=float, default=0.2, help="vuelo corto = microsalto [s]")
    a = p.parse_args()

    from loco_mujoco.trajectory.dataclasses import Trajectory
    from simulation.analysis.debug_agent import rollout

    agent_conf, agent_state, label, reference = load_agent(a.agent)
    rp = agent_conf.config.experiment.env_params.get("reward_params", {}) or {}
    ref_frac = float(rp.get("contact_ref_frac", 0.3))
    model = mujoco.MjModel.from_xml_path(str(C.XML_PATH))
    out = dict(agent=label, reference=str(reference), short_s=a.short, contact_ref_frac=ref_frac, legs=LEGS)

    # ---- referencia completa
    traj = Trajectory.load(str(reference))
    rq, rv = np.asarray(traj.data.qpos), np.asarray(traj.data.qvel)
    ref_dt = 1.0 / float(traj.info.frequency)
    rz, rvel = foot_kinematics(model, rq, rv)
    rc = contacts(rz)
    rlab = reference_labels(rz, ref_frac)
    # la referencia es cinematica: sus pies no siempre bajan hasta el suelo (los delanteros quedan a
    # ~6 cm), asi que sus vuelos y su ciclo se miden con la etiqueta relativa de MimicReward
    cycle_s = reference_cycle_s(rlab, ref_dt)
    out["reference_foot_min_height_cm"] = {leg: float(100 * rz[:, i].min()) for i, leg in enumerate(LEGS)}
    out["reference_cycle_s"] = cycle_s
    out["reference_stats"] = {}
    for i, leg in enumerate(LEGS):
        s = foot_stats(rz[:, i], rvel[:, i, 2], rlab[:, i], ref_dt, a.short, cycle_s)
        s["mimic_label_stance_frac"] = float(rlab[:, i].mean())
        s["mimic_label_vs_height_agree_pct"] = float(100 * (rlab[:, i] == rc[:, i]).mean())
        out["reference_stats"][leg] = s

    # ---- agente, en plano
    env = make_env(agent_conf, reference, 0.0, "up")
    dt = float(env.dt)
    steps = int(a.seconds / dt)
    warm = int(a.warmup / dt)
    act_low, act_high = np.asarray(env.info.action_space.low), np.asarray(env.info.action_space.high)
    for mode in ("deterministic", "stochastic"):
        d = rollout(env, agent_conf, agent_state, a.n_envs, steps, deterministic=mode == "deterministic",
                    seed=0)
        done = d["done"].astype(bool)
        fell = d["absorbing"].astype(bool)
        legs = {leg: [] for leg in LEGS}
        vz_trunk, agree = [], [[] for _ in LEGS]
        sat = np.zeros(len(JOINTS))
        n_sat = 0
        for e in range(a.n_envs):
            # tramo continuo: tras el calentamiento y hasta el primer reinicio (caida u horizonte)
            end = int(done[:, e].argmax()) if done[:, e].any() else steps
            if end - warm < int(2 * cycle_s / dt):
                continue
            q, qd = d["qpos"][warm:end, e], d["qvel"][warm:end, e]
            z, v = foot_kinematics(model, q, qd)
            c = contacts(z)
            ok = np.ones(len(q), bool)
            jump = np.abs(np.diff(q[:, 0])) > 0.1                 # salto de vuelta de la referencia
            ok[1:] &= ~jump
            ok[:-1] &= ~jump
            rz_e, _ = foot_kinematics(model, d["ref_qpos"][warm:end, e], d["ref_qvel"][warm:end, e])
            lab = rz_e <= rz.min(0) + ref_frac * (rz.max(0) - rz.min(0))
            for i, leg in enumerate(LEGS):
                legs[leg].append((z[:, i], v[:, i, 2], c[:, i], ok))
                agree[i].append((c[:, i] == lab[:, i])[ok])
            vz_trunk.append(qd[ok, 2])
            act = d["action"][warm:end, e]
            tol = 0.02 * (act_high - act_low)
            sat += ((act <= act_low + tol) | (act >= act_high - tol)).sum(0)
            n_sat += len(act)
        res = dict(n_envs_used=len(vz_trunk), fall_frac=float(fell.any(0).mean()),
                   trunk_vz_rms_mps=float(np.sqrt(np.mean(np.concatenate(vz_trunk) ** 2))),
                   action_saturation_pct={j: float(100 * s / max(n_sat, 1)) for j, s in zip(JOINTS, sat)},
                   legs={})
        for i, leg in enumerate(LEGS):
            # concatenar envs con un paso no valido entre medio para no unir vuelos de envs distintas
            parts = legs[leg]
            sep = lambda k: [np.r_[x[k], x[k][-1:]] for x in parts]
            zc, vc, cc = (np.concatenate(sep(k)) for k in range(3))
            okc = np.concatenate([np.r_[x[3], False] for x in parts])
            s = foot_stats(zc, vc, cc, dt, a.short, cycle_s, okc)
            s["agree_with_mimic_label_pct"] = float(100 * np.concatenate(agree[i]).mean())
            res["legs"][leg] = s
        out[mode] = res

    # ---- resumen
    print("altura minima del pie en la referencia [cm]: " +
          ", ".join(f"{k} {v:.1f}" for k, v in out["reference_foot_min_height_cm"].items()))
    print(f"\nAgente {label} | referencia {Path(str(reference)).name} | ciclo de la referencia {cycle_s:.3f} s")
    hdr = (f"{'':14s}{'pata':>5s}{'vuelos':>8s}{'med[s]':>8s}{'<0.1s%':>8s}{f'<{a.short}s%':>8s}"
           f"{'alt.corto[cm]':>14s}{'piso/ciclo':>11s}{'vz_td':>7s}{'apoyo':>7s}{'=etiq%':>8s}")
    print(hdr)
    rows = [("referencia", out["reference_stats"])] + [(m, out[m]["legs"]) for m in ("deterministic", "stochastic")]
    f = lambda x, fmt: (format(x, fmt) if x is not None else "-".rjust(int(fmt.split(".")[0])))
    for name, st in rows:
        for leg in LEGS:
            s = st[leg]
            ag = s.get("agree_with_mimic_label_pct", s.get("mimic_label_vs_height_agree_pct"))
            print(f"{name:14s}{leg:>5s}{s['n_flights']:8d}{f(s['flight_median_s'], '8.3f')}"
                  f"{f(s['pct_flights_lt_0p1s'], '8.1f')}{f(s['pct_flights_lt_short'], '8.1f')}"
                  f"{f(s['short_flight_peak_cm'], '14.2f')}{f(s['touchdowns_per_cycle'], '11.2f')}"
                  f"{f(s['touchdown_vz_median_mps'], '7.2f')}{s['stance_frac']:7.2f}{f(ag, '8.1f')}")
    for m in ("deterministic", "stochastic"):
        r = out[m]
        print(f"\n{m}: envs {r['n_envs_used']}, caidas {r['fall_frac']:.0%}, vz tronco rms {r['trunk_vz_rms_mps']:.2f} m/s")
        print("  saturacion de acciones [%]: " +
              ", ".join(f"{j} {v:.1f}" for j, v in r["action_saturation_pct"].items()))
    path = ROOT / f"hop_diagnosis_{label}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
