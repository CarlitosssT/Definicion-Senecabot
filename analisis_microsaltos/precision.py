"""Imitation accuracy of an agent vs its reference (deterministic, surviving envs).
usage (from seneca_loco/): python ../analisis_microsaltos/precision.py AGENT.pkl REF.npz TAG"""
import sys, numpy as np, mujoco
from omegaconf import OmegaConf
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.trajectory.dataclasses import Trajectory
from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf
from simulation.analysis.debug_agent import rollout

agent, ref, tag = sys.argv[1:4]
ac, as_ = PPOJax.load_agent(agent)
cfg = OmegaConf.to_container(ac.config.experiment.env_params, resolve=True); cfg["headless"] = True
env = TaskFactory.get_factory_cls("ImitationFactory").make(**cfg, custom_dataset_conf=CustomDatasetConf(traj=Trajectory.load(ref)))
m = env.get_model(); dd = mujoco.MjData(m)
N, T = 32, 800
d = rollout(env, ac, as_, N, T, deterministic=True, seed=0)
done = d["done"].astype(bool); envs = np.where(~done.any(0))[0]
q, rq = d["qpos"][100:, envs], d["ref_qpos"][100:, envs]
dj = np.degrees(q[..., 7:] - rq[..., 7:])
legs = {"FR": slice(0, 3), "FL": slice(3, 6), "BR": slice(6, 9), "BL": slice(9, 12)}
names = ["hip", "knee", "ankle"]
print(f"\n=== {tag}: surv {len(envs)}/{N}")
print("joint RMS [deg] total %.2f | " % np.sqrt((dj**2).mean()) + "  ".join(f"{k} {np.sqrt((dj[..., s]**2).mean()):.1f}" for k, s in legs.items())
      + " | by type " + "  ".join(f"{n} {np.sqrt((dj[..., i::3]**2).mean()):.1f}" for i, n in enumerate(names)))
print(f"root height err RMS {np.sqrt(((q[...,2]-rq[...,2])**2).mean())*100:.2f} cm (agent mean {q[...,2].mean()*100:.1f}, ref {rq[...,2].mean()*100:.1f})")
def tilt(qq):
    w, x, y, z = np.moveaxis(qq[..., 3:7], -1, 0)
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1))); roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    return pitch, roll
p, r_ = tilt(q); print(f"body pitch RMS {np.sqrt((p**2).mean()):.2f} deg, roll RMS {np.sqrt((r_**2).mean()):.2f} deg (ref is level)")
sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in ("fr_foot_mimic", "fl_foot_mimic", "br_foot_mimic", "bl_foot_mimic")]
def feet_rel(qs):
    out = []
    for qi in qs:
        dd.qpos[:] = qi; dd.qpos[0:3] = 0; dd.qpos[3:7] = [1, 0, 0, 0]; mujoco.mj_kinematics(m, dd); out.append(dd.site_xpos[sid].copy())
    return np.array(out)
sub = (slice(None, None, 5), slice(None))
fa, fr = feet_rel(q[sub].reshape(-1, q.shape[-1])), feet_rel(rq[sub].reshape(-1, q.shape[-1]))
err = np.linalg.norm(fa - fr, axis=-1) * 100
print("foot pos rel. body err [cm] mean " + f"{err.mean():.2f} | " + "  ".join(f"{k} {err[:, i].mean():.1f}" for i, k in enumerate(legs)))
vx = (d["qpos"][-1, envs, 0] - d["qpos"][100, envs, 0]) / ((T-101)*0.01); rvx = (d["ref_qpos"][-1, envs, 0] - d["ref_qpos"][100, envs, 0]) / ((T-101)*0.01)
print(f"speed agent {vx.mean():.3f} ± {vx.std():.3f} m/s | ref {rvx.mean():.3f} m/s | ratio {vx.mean()/rvx.mean():.2f}")
