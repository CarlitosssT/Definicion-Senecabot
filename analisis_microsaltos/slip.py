"""Foot slip while in ground contact (world-frame horizontal foot speed during stance).
usage (from seneca_loco/): python ../analisis_microsaltos/slip.py AGENT.pkl REF.npz TAG"""
import sys, numpy as np, mujoco
from omegaconf import OmegaConf
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.trajectory.dataclasses import Trajectory
from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf
from simulation.analysis.debug_agent import rollout, FOOT_GEOM_NAMES
from simulation.analysis.gait_visuals import CONTACT_TOL

agent, ref, tag = sys.argv[1:4]
ac, as_ = PPOJax.load_agent(agent)
cfg = OmegaConf.to_container(ac.config.experiment.env_params, resolve=True); cfg["headless"] = True
env = TaskFactory.get_factory_cls("ImitationFactory").make(**cfg, custom_dataset_conf=CustomDatasetConf(traj=Trajectory.load(ref)))
m = env.get_model(); dd = mujoco.MjData(m)
fid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOM_NAMES]
N, T = 16, 500
d = rollout(env, ac, as_, N, T, deterministic=True, seed=0)
envs = np.where(~d["done"].astype(bool).any(0))[0]
pos = np.zeros((T, len(envs), 4, 3))
for j, e in enumerate(envs):
    for t in range(T):
        dd.qpos[:] = d["qpos"][t, e]; mujoco.mj_kinematics(m, dd); pos[t, j] = dd.geom_xpos[fid]
pos = pos[100:]
v = np.linalg.norm(np.diff(pos[..., :2], axis=0), axis=-1) / 0.01          # horizontal foot speed [m/s]
contact = (pos[1:, ..., 2] - m.geom_size[fid, 0]) <= CONTACT_TOL
body_v = (d["qpos"][-1, envs, 0] - d["qpos"][100, envs, 0]) / ((T - 101) * 0.01)
sl = [v[..., k][contact[..., k]] for k in range(4)]
allv = np.concatenate(sl)
print(f"\n=== {tag}: body {body_v.mean():.3f} m/s")
print("foot speed while in contact [cm/s] mean: " + "  ".join(f"{n} {s.mean()*100:.0f}" for n, s in zip(["FR", "FL", "BR", "BL"], sl))
      + f" | all {allv.mean()*100:.0f} (median {np.median(allv)*100:.0f}) | slip/body speed {allv.mean()/body_v.mean():.2f}")
