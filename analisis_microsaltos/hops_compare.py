"""Micro-hop metrics for an agent, evaluated on a given reference trajectory.
usage (from seneca_loco/): python ../analisis_microsaltos/hops_compare.py AGENT.pkl REF.npz TAG FIG.png"""
import sys
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from scipy.signal import find_peaks
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.trajectory.dataclasses import Trajectory
from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf
from simulation.analysis.debug_agent import rollout, FOOT_GEOM_NAMES
from simulation.analysis.gait_visuals import _cycle_troughs, CONTACT_TOL

agent, ref, tag, fig_path = sys.argv[1:5]
DT, LEGS, N, T = 0.01, ["FR", "FL", "BR", "BL"], 32, 800
ac, as_ = PPOJax.load_agent(agent)
cfg = OmegaConf.to_container(ac.config.experiment.env_params, resolve=True); cfg["headless"] = True
env = TaskFactory.get_factory_cls("ImitationFactory").make(**cfg, custom_dataset_conf=CustomDatasetConf(traj=Trajectory.load(ref)))
m = env.get_model()
fid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOM_NAMES]
r = m.geom_size[fid, 0]
td = env.th.traj.data
sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in ("fr_foot_mimic", "fl_foot_mimic", "br_foot_mimic", "bl_foot_mimic")]
rz = np.asarray(td.site_xpos)[:, sid, 2]
lo, hi = np.percentile(rz, 2, 0), np.percentile(rz, 98, 0)
ref_c = rz <= lo + 0.3 * (hi - lo)

d = rollout(env, ac, as_, N, T, deterministic=True, seed=0, collect_dynamics=True)
done = d["done"].astype(bool)
envs = np.where(~done.any(0))[0]
z = d["foot_z"] - r[None, None, :]                    # foot bottom height above floor
st = z <= CONTACT_TOL
rc = ref_c[np.asarray(td.split_points)[0] + d["subtraj_step_no"]]
td_n, cyc, short, dips, match, stance, lift = np.zeros(4), 0, np.zeros(4), np.zeros(4), [], [], []
for e in envs:
    tr = _cycle_troughs(d["ref_qpos"][:, e, 7])
    if len(tr) < 2: continue
    a, b = tr[0], tr[-1]; cyc += len(tr) - 1
    s = st[a:b, e]
    td_n += np.sum(np.diff(s.astype(int), axis=0) == 1, axis=0)
    for k in range(4):
        mm = np.concatenate([[0], s[:, k].astype(int), [0]]); dd = np.diff(mm)
        short[k] += np.sum((np.where(dd == -1)[0] - np.where(dd == 1)[0]) <= 5)
        pk, _ = find_peaks(-z[a:b, e, k], prominence=0.005)
        dips[k] += np.sum(z[a:b, e, k][pk] > CONTACT_TOL)
    match.append((s == rc[a:b, e]).mean(0)); stance.append(s.mean(0))
    lift.append(np.percentile(z[a:b, e], 95, axis=0))
q = d["qpos"]; vx = (q[-1, envs, 0] - q[100, envs, 0]) / ((T - 101) * DT)
rvx = (d["ref_qpos"][-1, envs, 0] - d["ref_qpos"][100, envs, 0]) / ((T - 101) * DT)
a = d["action"][:, envs]
jerk = np.sum((a[2:] - 2 * a[1:-1] + a[:-2]) ** 2, -1).mean()
print(f"\n=== {tag}: surv {len(envs)}/{N}, {cyc} cycles")
print(f"speed {vx.mean():.3f} m/s (ref {rvx.mean():.3f}, ratio {vx.mean()/rvx.mean():.2f}) | action jerk {jerk:.3f}")
print(f"{'':4s}{'touchdowns/cyc':>15s}{'short<=50ms/cyc':>16s}{'air dips/cyc':>13s}{'stance frac':>12s}{'ref stance':>11s}{'contact match':>14s}{'lift p95 cm':>12s}")
M, S, L = np.mean(match, 0), np.mean(stance, 0), np.mean(lift, 0) * 100
for k, n in enumerate(LEGS):
    print(f"{n:4s}{td_n[k]/cyc:15.2f}{short[k]/cyc:16.2f}{dips[k]/cyc:13.2f}{S[k]:12.2f}{ref_c[:, k].mean():11.2f}{M[k]:14.2f}{L[k]:12.1f}")
print(f"TOTAL touchdowns/cyc {td_n.sum()/cyc:.2f} | mean contact match {M.mean():.2f}")

e = envs[0]; t = np.arange(T) * DT; sl = slice(100, 400)
f, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
for k, n in enumerate(LEGS):
    ax[k].plot(t[sl], z[sl, e, k] * 100, color="tab:blue", label="agent foot height")
    ax[k].fill_between(t[sl], 0, rc[sl, e, k] * 8, color="tab:orange", alpha=0.25, step="mid", label="reference stance")
    ax[k].axhline(CONTACT_TOL * 100, color="gray", ls=":", lw=0.8); ax[k].set_ylabel(f"{n} [cm]")
ax[0].legend(loc="upper right", fontsize=8); ax[-1].set_xlabel("time [s]"); f.suptitle(f"Foot height — {tag}")
f.tight_layout(); f.savefig(fig_path, dpi=110); print("fig:", fig_path)
