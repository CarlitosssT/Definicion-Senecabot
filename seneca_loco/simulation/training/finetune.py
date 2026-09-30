"""
finetune.py — start a PPO training from an already trained agent instead of from scratch.

Used by train.py when the Hydra config has a ``finetune`` block (e.g. conf_finetune_slope.yaml):

  * resolve_config(): the experiment config becomes the init agent's OWN saved one (same reward,
    network and PPO settings it was trained with) merged with ``finetune.experiment`` (new
    total_timesteps, lr, env additions such as the slope randomizer and gravity observation).
  * init_agent_state(): the new network starts with the init agent's weights and observation
    normalization. Observation entries the init agent did not have (e.g. the trunk-frame gravity
    added by ``gravity_obs``) get ZERO input weights in the actor and the critic, so the starting
    policy and value function are exactly the init agent's (checked numerically); their
    normalization statistics are measured by running the init policy in the new env. The optimizer
    state is new (fresh Adam moments and learning-rate schedule).
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf, open_dict

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax, TrainState
from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf
from loco_mujoco.trajectory.dataclasses import Trajectory

from simulation.config import paths

# recomputed by PPOJax.init_agent_conf from total_timesteps / num_envs / num_steps
_DERIVED_KEYS = ("num_updates", "minibatch_size", "validation_interval")
_INPUT_LAYERS = (("FullyConnectedNet_0", "Dense_0"),     # actor
                 ("FullyConnectedNet_1", "Dense_0"))     # critic


def _path(p) -> Path:
    """Paths in the fine-tune config are relative to the seneca_loco root unless absolute."""
    p = Path(str(p))
    return p if p.is_absolute() else paths.PROJECT_ROOT / p


def reference_path(config) -> Path:
    return _path(config.finetune.reference)


def resolve_config(config):
    """Replace ``config.experiment`` by the init agent's saved experiment config merged with
    ``config.finetune.experiment``."""
    ft = config.finetune
    init_conf, _ = PPOJax.load_agent(str(_path(ft.init_agent)))
    exp = OmegaConf.to_container(init_conf.config.experiment, resolve=True)
    for k in _DERIVED_KEYS:
        exp.pop(k, None)
    # a component whose type changes (e.g. MimicReward -> SlopeLocomotionReward) drops the init agent's
    # parameters of the old type instead of merging them into the new one
    new_env = ft.experiment.get("env_params") or {}
    for kind in ("reward", "terminal_state", "domain_randomization", "init_state", "goal", "control"):
        new_type = new_env.get(f"{kind}_type")
        if new_type is not None and new_type != exp["env_params"].get(f"{kind}_type"):
            exp["env_params"].pop(f"{kind}_params", None)
    exp = OmegaConf.merge(OmegaConf.create(exp), ft.experiment)
    assert exp.n_seeds == 1, "Fine-tuning supports a single seed."

    # slope curriculum given as a fraction of this training -> steps per env (what SlopeRandomizer counts)
    if ft.get("curriculum_frac") and exp.env_params.get("domain_randomization_type") == "SlopeRandomizer":
        steps = int(ft.curriculum_frac * exp.total_timesteps / exp.num_envs)
        exp = OmegaConf.merge(exp, {"env_params": {"domain_randomization_params": {"curriculum_steps": steps}}})

    with open_dict(config):
        config.experiment = exp
    return config


def _make_env(env_params, reference):
    cfg = OmegaConf.to_container(env_params, resolve=True)
    cfg["headless"] = True
    return TaskFactory.get_factory_cls("ImitationFactory").make(
        **cfg, custom_dataset_conf=CustomDatasetConf(traj=Trajectory.load(str(reference))))


def init_agent_state(config, env, agent_conf, n_envs=256, n_steps=200, seed=0):
    """Fine-tune start state: the init agent's weights and normalization, mapped onto the new
    observation layout by observation name (see module docstring)."""
    ft = config.finetune
    old_conf, old_state = PPOJax.load_agent(str(_path(ft.init_agent)))
    old_ts = old_state.train_state

    # ---- observation layout: old -> new, matched by observation name ---------------------------
    old_env = _make_env(old_conf.config.experiment.env_params, reference_path(config))
    old_ind = {k: np.asarray(o.obs_ind) for k, o in old_env.obs_container.items()}
    new_ind = {k: np.asarray(o.obs_ind) for k, o in env.obs_container.items()}
    missing = set(old_ind) - set(new_ind)
    assert not missing, f"The new env lacks observations of the init agent: {sorted(missing)}"
    for k in old_ind:
        assert len(old_ind[k]) == len(new_ind[k]), f"Observation {k} changed size."
    old_dim = old_env.info.observation_space.shape[0]
    new_dim = env.info.observation_space.shape[0]
    src = np.concatenate([old_ind[k] for k in old_ind])
    dst = np.concatenate([new_ind[k] for k in old_ind])
    gather = np.empty(old_dim, dtype=int)
    gather[src] = dst                                         # old_obs = new_obs[..., gather]
    added = np.setdiff1d(np.arange(new_dim), dst)
    added_names = [k for k in new_ind if k not in old_ind]
    print(f"[finetune] obs {old_dim} -> {new_dim}; new entries {added_names} at {added.tolist()}")

    # ---- weights: copy, zero input rows for the new entries -----------------------------------
    params = jax.tree.map(lambda x: np.array(x, dtype=np.float32), old_ts.params)
    for net, layer in _INPUT_LAYERS:
        k_old = params[net][layer]["kernel"]
        assert k_old.shape[0] == old_dim
        k_new = np.zeros((new_dim, k_old.shape[1]), dtype=np.float32)
        k_new[dst] = k_old[src]
        params[net][layer]["kernel"] = k_new
    if ft.get("init_std") is not None:
        params["log_std"] = np.full_like(params["log_std"], np.log(float(ft.init_std)))

    ref_params = agent_conf.network.init(jax.random.key(0), jnp.zeros(new_dim))["params"]
    shapes = lambda t: jax.tree.map(lambda x: tuple(np.shape(x)), t)
    assert shapes(ref_params) == shapes(params), "Network architecture differs from the init agent's."

    # ---- normalization of the new entries, measured with the init policy ----------------------
    old_rs = old_ts.run_stats["RunningMeanStd_0"]
    old_vars = {"params": old_ts.params, "run_stats": old_ts.run_stats}

    def old_policy(obs):
        (pi, value), _ = old_conf.network.apply(old_vars, obs[..., gather], mutable=["run_stats"])
        return pi.mean(), value

    def step(state, _):
        state = jax.vmap(env.mjx_step)(state, old_policy(state.observation)[0])
        return state, state.observation

    state = jax.vmap(env.mjx_reset)(jax.random.split(jax.random.key(seed), n_envs))
    state, obs_seq = jax.jit(lambda s: jax.lax.scan(step, s, None, length=n_steps))(state)
    obs_added = np.asarray(obs_seq)[..., added].reshape(-1, len(added))

    mean = np.zeros(new_dim, dtype=np.float32)
    var = np.ones(new_dim, dtype=np.float32)
    mean[dst], var[dst] = np.asarray(old_rs["mean"])[src], np.asarray(old_rs["var"])[src]
    mean[added] = obs_added.mean(0)
    var[added] = np.maximum(obs_added.var(0), 1e-4)
    run_stats = {"RunningMeanStd_0": {"mean": mean, "var": var,
                                      "count": np.asarray(old_rs["count"], dtype=np.float32)}}
    print(f"[finetune] new entries normalization: mean {np.round(mean[added], 3).tolist()}, "
          f"std {np.round(np.sqrt(var[added]), 3).tolist()}")

    # ---- check: the start policy and value ARE the init agent's --------------------------------
    obs = np.asarray(obs_seq[-1])
    a_old, v_old = old_policy(obs)
    (pi_new, v_new), _ = agent_conf.network.apply({"params": params, "run_stats": run_stats}, obs,
                                                  mutable=["run_stats"])
    err = max(float(jnp.abs(a_old - pi_new.mean()).max()), float(jnp.abs(v_old - v_new).max()))
    assert err < 1e-4, f"Start policy differs from the init agent (max |diff| = {err:.2e})."
    print(f"[finetune] start policy/value == init agent's (max |diff| {err:.1e})")

    train_state = TrainState.create(apply_fn=agent_conf.network.apply, params=params,
                                    run_stats=run_stats, tx=agent_conf.tx)
    return PPOJax._agent_state(train_state=train_state)
