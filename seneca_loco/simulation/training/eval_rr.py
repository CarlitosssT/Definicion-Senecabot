"""
eval_rr.py — headless evaluation of an agent trained on the RR model (conf_rr.yaml).

    python simulation/training/eval_rr.py --path <PPOJax_saved.pkl> [--n_envs 32] [--n_steps 1000] [--video]

Runs the deterministic policy on n_envs MjWarp envs started from random reference frames and reports, against the
reference gait: falls (absorbing states), forward speed of the trunk, RMS error of knee and ankle of every leg, trunk
height, and the reward per step. With --video it also records one env (LocoMuJoCo_recordings/).
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from omegaconf import OmegaConf

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf
from loco_mujoco.trajectory.dataclasses import Trajectory

from simulation.config import paths

LEGS = ("fr", "fl", "br", "bl")


def _make_env(config, n_envs, headless=True, camera=None, reference=None):
    OmegaConf.set_struct(config, False)
    ep = config.experiment.env_params
    ep["headless"] = headless
    ep["record_camera"] = camera
    ep["nconmax"] = max(64, 13 * n_envs)
    ref = config.get("reference")
    if config.get("finetune") is not None and config.finetune.get("reference") is not None:
        ref = config.finetune.reference                  # fine-tuned agents keep it in the finetune block
    if reference is not None:
        ref = reference
    traj = Trajectory.load(paths.PROJECT_ROOT / str(ref) if ref is not None else paths.TRAJ_ADAPTED)
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    return factory.make(**ep, custom_dataset_conf=CustomDatasetConf(traj=traj))


def _inner(state):
    while hasattr(state, "env_state"):
        state = state.env_state
    return state


def evaluate(path, n_envs=32, n_steps=1000, seed=0, reference=None):
    agent_conf, agent_state = PPOJax.load_agent(str(path))
    config = agent_conf.config
    env = _make_env(config, n_envs, reference=reference)
    model = env.model
    jid = {f"{l}_{j}": model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{l}_{j}_joint")]
           for l in LEGS for j in ("knee", "ankle")}
    names, cols = list(jid), np.array(list(jid.values()))
    dcols = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_joint")] for n in names])
    feet = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{l}_foot_mimic") for l in LEGS])

    wenv = PPOJax._wrap_env(env, config.experiment)
    ts = agent_state.train_state
    ts.params["log_std"] = np.ones_like(ts.params["log_std"]) * -np.inf      # deterministic

    def act(ts, obs, key):
        (pi, _), upd = agent_conf.network.apply({"params": ts.params, "run_stats": ts.run_stats}, obs,
                                                mutable=["run_stats"])
        return pi.sample(seed=key), ts.replace(run_stats=upd["run_stats"])

    dt = float(env.dt)
    traj_qpos = env.th.traj.data.qpos
    split = env.th.traj.data.split_points

    @jax.jit
    def step(carry, key):
        ts, obs, st = carry
        a, ts = act(ts, obs, key)
        obs, rew, absorbing, done, info, st = wenv.step(st, a)
        s = _inner(st)
        tr = s.additional_carry.traj_state
        ref = traj_qpos[split[tr.traj_no] + tr.subtraj_step_no]
        out = dict(a=jnp.clip(a, -1, 1), dq=s.data.qvel[:, dcols], foot_z=s.data.site_xpos[:, feet, 2], q=s.data.qpos[:, cols], ref=ref[:, cols], x=s.data.qpos[:, 0], z=s.data.qpos[:, 2], z_ref=ref[:, 2],
                   vx=s.data.qvel[:, 0], ref_vx=env.th.traj.data.qvel[split[tr.traj_no] + tr.subtraj_step_no][:, 0],
                   rew=s.reward, absorbing=absorbing, done=done)
        return (ts, obs, st), out

    rng = jax.random.key(seed)
    rng, *keys = jax.random.split(rng, n_envs + 1)
    obs, st = wenv.reset(jnp.stack(keys))
    carry, outs = (ts, obs, st), []
    for k in jax.random.split(rng, n_steps):
        carry, o = step(carry, k)
        outs.append(jax.device_get(o))
    o = {k: np.stack([x[k] for x in outs]) for k in outs[0]}          # [T, n_envs, ...]

    alive = np.cumprod(~o["absorbing"].astype(bool), axis=0).astype(bool)   # until the first fall of each env
    err = np.where(alive[..., None], o["q"] - o["ref"], np.nan)
    rms = {n: float(np.degrees(np.sqrt(np.nanmean(err[..., i] ** 2)))) for i, n in enumerate(names)}
    vx = np.where(alive, o["vx"], np.nan)
    res = dict(
        n_envs=n_envs, n_steps=n_steps,
        falls=int((o["absorbing"].astype(bool) & np.vstack([np.ones((1, n_envs), bool), alive[:-1]])).sum()),
        envs_fallen=int((~alive[-1]).sum()),
        mean_steps_alive=float(alive.sum(0).mean()),
        speed_mean=float(np.nanmean(vx)), speed_ref=float(np.mean(o["ref_vx"])),
        z_mean=float(np.nanmean(np.where(alive, o["z"], np.nan))),
        reward_per_step=float(np.nanmean(np.where(alive, o["rew"], np.nan))),
        bias_deg={n: float(np.degrees(np.nanmean(err[..., i]))) for i, n in enumerate(names)},
        ref_range_deg={n: float(np.degrees(np.ptp(o["ref"][..., i]))) for i, n in enumerate(names)},
        agent_range_deg={n: float(np.degrees(np.nanpercentile(o["q"][..., i], 99) - np.nanpercentile(o["q"][..., i], 1)))
                         for i, n in enumerate(names)},
        z_ref=float(np.nanmean(o["z_ref"])),
        # smoothness: action 1st / 2nd differences (sum over the 8 actions, per step), joint accelerations, and
        # touchdowns per second of every foot (foot site below 0.037 m, as the contact reward); the reference
        # cycle (0.635 s) has ~1.6 touchdowns/s per foot, more means micro-hops
        action_rate=float(np.nanmean(np.where(alive[1:], np.sum(np.diff(o["a"], axis=0) ** 2, -1), np.nan))),
        action_jerk=float(np.nanmean(np.where(alive[2:], np.sum(np.diff(o["a"], 2, axis=0) ** 2, -1), np.nan))),
        qacc_rms={n: float(np.sqrt(np.nanmean(np.where(alive[1:], (np.diff(o["dq"][..., i], axis=0) / dt) ** 2, np.nan))))
                  for i, n in enumerate(names)},
        touchdowns_per_s={l: float(np.sum(np.diff((o["foot_z"][..., k] < 0.037).astype(int), axis=0) == 1)
                                   / (alive.sum() * dt)) for k, l in enumerate(LEGS)},
        rms_deg=rms, rms_deg_mean=float(np.mean(list(rms.values()))),
        # crank motors: mean |action| (1 = full ctrlrange) and fraction of steps at the limit, per actuator
        act_abs_mean={n: float(np.nanmean(np.where(alive, np.abs(o["a"][..., i]), np.nan)))
                      for i, n in enumerate([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, k)
                                             for k in range(model.nu)])},
        act_saturated=float(np.nanmean(np.where(alive[..., None], np.abs(o["a"]) > 0.98, np.nan))),
    )
    return res


def rollout_qpos(path, n_steps=300, reference=None):
    """Deterministic rollout of one env started at the first frame of the reference (no random start).
    Returns dict(qpos=[T, nq] agent, qpos_ref=[T, nq] reference at the same steps, dt, absorbing=[T])."""
    agent_conf, agent_state = PPOJax.load_agent(str(path))
    config = agent_conf.config
    env = _make_env(config, 1, reference=reference)
    env.th.random_start = False
    wenv = PPOJax._wrap_env(env, config.experiment)
    ts = agent_state.train_state
    ts.params["log_std"] = np.ones_like(ts.params["log_std"]) * -np.inf
    traj_qpos, split = env.th.traj.data.qpos, env.th.traj.data.split_points

    @jax.jit
    def step(carry, key):
        ts, obs, st = carry
        (pi, _), upd = agent_conf.network.apply({"params": ts.params, "run_stats": ts.run_stats}, obs,
                                                mutable=["run_stats"])
        ts = ts.replace(run_stats=upd["run_stats"])
        obs, rew, absorbing, done, info, st = wenv.step(st, jnp.reshape(pi.sample(seed=key), (1, -1)))
        s = _inner(st)
        tr = s.additional_carry.traj_state
        return (ts, obs, st), (s.data.qpos[0], traj_qpos[split[tr.traj_no] + tr.subtraj_step_no][0], absorbing[0])

    rng, k0 = jax.random.split(jax.random.key(0))
    obs, st = wenv.reset(k0[None])
    carry, q, qr, ab = (ts, obs, st), [], [], []
    for k in jax.random.split(rng, n_steps):
        carry, (a, b, c) = step(carry, k)
        q.append(np.asarray(a)); qr.append(np.asarray(b)); ab.append(bool(c))
    return dict(qpos=np.stack(q), qpos_ref=np.stack(qr), dt=float(env.dt), absorbing=np.array(ab))


def record(path, n_steps=1000):
    agent_conf, agent_state = PPOJax.load_agent(str(path))
    env = _make_env(agent_conf.config, 1, headless=True, camera="follow")
    PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=n_steps, n_envs=1, record=True,
                       train_state_seed=0)
    return env.video_file_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--n_envs", type=int, default=32)
    ap.add_argument("--n_steps", type=int, default=1000)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--reference", default=None, help="override the agent's reference (path relative to seneca_loco/)")
    args = ap.parse_args()
    res = evaluate(args.path, args.n_envs, args.n_steps, reference=args.reference)
    print(json.dumps(res, indent=2))
    (Path(args.path).parent / "eval_rr.json").write_text(json.dumps(res, indent=2))
    if args.video:
        print("video:", record(args.path, args.n_steps))
