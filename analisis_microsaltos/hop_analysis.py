"""Detect foot micro-hops: agent (true sim contact) vs reference (kinematic foot height)."""
import sys
from pathlib import Path
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.analysis.thesis_eval_figures import build_env
from simulation.analysis.debug_agent import rollout, FOOT_GEOM_NAMES
from simulation.analysis.gait_visuals import _cycle_troughs, _foot_z_from_qpos, CONTACT_TOL

DT = 0.01            # env control step (100 Hz)
OUT = Path(sys.argv[2])
LEGS = ["FR", "FL", "BR", "BL"]


def runs(mask):
    """Lengths of consecutive True runs per column -> list of arrays."""
    out = []
    for c in range(mask.shape[1]):
        m = np.concatenate([[0], mask[:, c].astype(int), [0]])
        d = np.diff(m)
        s, e = np.where(d == 1)[0], np.where(d == -1)[0]
        out.append(e - s)
    return out


def swing_minima(z, lift):
    """Count local minima of foot height that occur while the foot is 'lifted' (> lift):
    a dip mid-swing that doesn't reach the ground = a stutter / micro-hop in the air."""
    from scipy.signal import find_peaks
    cnt = []
    for c in range(z.shape[1]):
        pk, _ = find_peaks(-z[:, c], prominence=0.005)
        cnt.append(int(np.sum(z[pk, c] > lift[c])))
    return cnt


path = sys.argv[1]
env, ac, as_, cfg = build_env(path)
model = env.get_model()
fid = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOM_NAMES])
fr = np.array([float(model.geom_size[f][0]) for f in fid])

d = rollout(env, ac, as_, 16, 600, deterministic=True, seed=0, collect_dynamics=True)
done = d["done"].astype(bool)
surv = ~done.any(axis=0)
envs = np.where(surv)[0]
print(f"surviving envs: {len(envs)}/16")

rows = []
for tag in ("agent", "reference"):
    tot_td, tot_cyc, short, all_c, air_dips = np.zeros(4), 0, np.zeros(4), [[] for _ in range(4)], np.zeros(4)
    for e in envs:
        qpos = np.asarray(d["qpos" if tag == "agent" else "ref_qpos"])[:, e, :]
        if tag == "agent":
            z = np.asarray(d["foot_z"])[:, e, :]
            stance = z <= fr[None, :] + CONTACT_TOL
            lift = fr + CONTACT_TOL
        else:
            z = _foot_z_from_qpos(model, qpos, fid)
            lo, hi = np.percentile(z, 2, 0), np.percentile(z, 98, 0)
            thr = lo + 0.30 * (hi - lo)
            stance = z <= thr[None, :]
            lift = thr
        tr = _cycle_troughs(qpos[:, 7])
        if len(tr) < 2:
            continue
        a, b = tr[0], tr[-1]
        st = stance[a:b]
        tot_cyc += len(tr) - 1
        tot_td += np.sum(np.diff(st.astype(int), axis=0) == 1, axis=0)
        for i, r in enumerate(runs(st)):
            all_c[i] += list(r)
            short[i] += np.sum(r <= 5)          # contacts lasting <= 50 ms
        air_dips += swing_minima(z[a:b], lift)
    print(f"\n== {tag}: {tot_cyc} gait cycles")
    print(f"{'leg':4s} {'touchdowns/cycle':>17s} {'contacts<=50ms/cycle':>21s} {'median contact [ms]':>20s} {'air dips/cycle':>15s}")
    for i, L in enumerate(LEGS):
        med = np.median(all_c[i]) * DT * 1000 if all_c[i] else float("nan")
        print(f"{L:4s} {tot_td[i]/tot_cyc:17.2f} {short[i]/tot_cyc:21.2f} {med:20.0f} {air_dips[i]/tot_cyc:15.2f}")

# figure: foot heights of one surviving env, ~3 s, agent vs reference
e = envs[0] if len(envs) else 0
za = np.asarray(d["foot_z"])[:, e, :] - fr[None, :]
zr = _foot_z_from_qpos(model, np.asarray(d["ref_qpos"])[:, e, :], fid)
zr = zr - np.percentile(zr, 2, 0)[None, :]
t = np.arange(za.shape[0]) * DT
sl = slice(100, 400)
fig, axs = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
for i, L in enumerate(LEGS):
    axs[i].plot(t[sl], zr[sl, i] * 100, color="tab:orange", label="reference (kinematic, shifted to 0)")
    axs[i].plot(t[sl], za[sl, i] * 100, color="tab:blue", label="agent (height above floor)")
    axs[i].axhline(CONTACT_TOL * 100, color="gray", ls=":", lw=0.8)
    axs[i].set_ylabel(f"{L} foot [cm]")
axs[0].legend(loc="upper right", fontsize=8)
axs[-1].set_xlabel("time [s]")
fig.suptitle(f"Foot height — {Path(path).parent.name}")
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print(f"\nfigure: {OUT}")
