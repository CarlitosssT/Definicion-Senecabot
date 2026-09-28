"""
agent_diagnostics.py

Turn debug_agent.py's diagnostic rollout into a handful of *scalar* metrics that
score how well a trained policy imitates the reference — the same signals the
debug_agent figures show (survival, per-joint tracking, why robots die, action
smoothness), but reduced to numbers an HPO objective can minimize.

This is the bridge between `debug_agent.py` (visual diagnosis of ONE agent) and
`Optuna_optimization.ipynb` Phase 2 (search over reward params). The Phase-2
"Metric for Sweep" is built from `evaluate_agent(...)` here, so the sweep
optimizes exactly what the debug plots reveal. On the current agent
(2026-06-10/13-24-37, trained on the symmetrized reference) those are:
  - the **worst joints** (fl_ankle ≈ 16.7°, br_knee ≈ 11.9°, fl_knee ≈ 10.7°) that the
    validation EuclideanDistance, a single mean-over-joints scalar, cannot see;
  - a **crouched gait**: base height ≈ 0.33 m vs the reference's 0.40 m (~7 cm sink,
    held for the whole episode) — far inside the termination band, so nothing else
    penalizes it;
  - a **slow gait**: ~2.0 m traveled vs the reference's ~4.1 m over 3 s — HALF the
    reference pace (per-env speed ratio 0.46–0.54, so it's systematic, not noise).

It reuses debug_agent's *verified* rollout and termination logic (same
`rollout`, `first_done_index`, `read_terminal_thresholds`, layout constants), so
the numbers here match the figures byte-for-byte; only the reduction to scalars
is new.

Returned per-seed dict (all scalars unless noted):
    surv            fraction of robots that survive the whole horizon            (↑ better)
    eplen_frac      mean episode length / horizon                                (↑ better)
    joint_rms_mean  mean per-joint RMS tracking error [deg]                      (↓ better)
    joint_rms_worst mean of the WORST-k joints' RMS error [deg]                  (↓ better)
    joint_rms_max   single worst joint's RMS error [deg]                         (↓ better)
    per_joint_rms   (12,) per-joint RMS error [deg] — for inspection/logging
    bad_death_frac  fraction dying of instability / fall / tip (NOT benign       (↓ better)
                    clip-end wrap or horizon timeout)
    act_rate        mean |Δaction| per control step over alive steps             (↓ better)
    height_rms      RMS base-height error vs reference [m] over alive steps      (↓ better)
    speed_ratio     forward travel / reference's forward travel (1.0 = on pace)  (→1 better)
"""

import numpy as np

# reuse the *verified* rollout + termination machinery from debug_agent (single source of truth)
from simulation.analysis.debug_agent import (
    rollout, first_done_index, read_terminal_thresholds,
    JOINT_NAMES, QPOS_JOINT_SLICE,
)


def _classify_bad_deaths(data, thresholds, horizon, traj_len):
    """Fraction of envs that die of a *real* failure (instability / fall / tip), excluding the
    benign endings (clip-end wrap truncation, horizon timeout, survived). Replicates
    debug_agent.plot_termination_analysis's authoritative classifier: the recorded `absorbing`
    flag is the discriminator, and a non-absorbing pre-horizon `done` near the clip end is the
    intrinsic NaN-obs wrap (benign), not a sim blow-up."""
    done = data["done"].astype(bool)
    absb = data["absorbing"].astype(bool)
    T, N = done.shape
    died = done.any(axis=0)
    death_idx = first_done_index(done)                 # T if survived
    di = np.clip(death_idx, 0, T - 1)
    term_idx = np.clip(death_idx - 1, 0, T - 1)
    ar = np.arange(N)

    absorb_at_death = absb[di, ar] & died              # fell (height) / tipped (orientation)
    is_timeout = died & (~absorb_at_death) & (death_idx >= horizon - 1)
    nonabsorb_early = died & (~absorb_at_death) & (~is_timeout)
    sub_term = (data["subtraj_step_no"][term_idx, ar] if "subtraj_step_no" in data
                else np.zeros(N))
    is_wrap = nonabsorb_early & (sub_term >= max(traj_len - 3, 0))   # clip ran out -> benign
    is_instab = nonabsorb_early & (~is_wrap)                          # real blow-up

    bad = absorb_at_death | is_instab
    return float(bad.sum()) / max(N, 1)


def _scalars_for_one(data, thresholds, horizon, traj_len, worst_k):
    """Reduce one rollout's (T, n_envs, ...) arrays to the scalar metric dict above. `alive`
    masks each env from start through (and including) its terminating step, so dead-tail steps
    don't pollute the tracking error."""
    done = data["done"].astype(bool)
    T, N = done.shape
    alive = ((np.cumsum(done, axis=0) - done) == 0).astype(float)    # (T, N) 1 until first done

    # survival + episode length
    surv = float((~done.any(axis=0)).mean())
    eplen_frac = float(first_done_index(done).mean() / max(T, 1))

    # per-joint RMS tracking error [deg], over alive (env, step) samples
    err = np.rad2deg(data["qpos"][:, :, QPOS_JOINT_SLICE]
                     - data["ref_qpos"][:, :, QPOS_JOINT_SLICE])     # (T, N, 12)
    a = alive[..., None]
    per_joint = np.sqrt((err ** 2 * a).sum(axis=(0, 1)) / np.maximum(a.sum(), 1e-6))  # (12,)
    order = np.argsort(per_joint)[::-1]
    worst = per_joint[order[:worst_k]]

    # action smoothness: mean |Δaction| per step over alive transitions
    act = data["action"]                                            # (T, N, 12)
    dact = np.abs(np.diff(act, axis=0))                             # (T-1, N, 12)
    am = alive[1:][..., None]
    act_rate = float((dact * am).sum() / np.maximum(am.sum() * act.shape[-1], 1e-6))

    # base-height tracking [m]: RMS of (policy z - reference z). Uses a strict mask that
    # excludes each env's terminating step (the env resets on `done`, so data there is
    # post-reset). Catches the "crouched walk" failure (e.g. 0.33 m held vs 0.40 m target)
    # that sits far inside the termination band and is invisible to the joint metrics.
    strict = alive * (~done)
    hz_err = data["qpos"][:, :, 2] - data["ref_qpos"][:, :, 2]      # (T, N)
    height_rms = float(np.sqrt((hz_err ** 2 * strict).sum()
                               / np.maximum(strict.sum(), 1e-6)))

    # forward pace vs the reference: per-env xy displacement over the env's own alive window,
    # projected onto that env's reference displacement (so heading drift also reads as a
    # deficit). 1.0 = keeps pace; the current agent reads ~0.51 (HALF the reference pace).
    # World x/y is untracked (track_root_xy=False) but *displacement from own start* is
    # comparable (the same fix debug_agent's root_tracking plot uses). Envs with too short a
    # window or a ~static reference segment are excluded.
    end = np.clip(first_done_index(done) - 1, 0, T - 1)             # last pre-reset step
    ar = np.arange(N)
    pol_d = data["qpos"][end, ar, 0:2] - data["qpos"][0, :, 0:2]
    ref_d = data["ref_qpos"][end, ar, 0:2] - data["ref_qpos"][0, :, 0:2]
    ref_n = np.linalg.norm(ref_d, axis=-1)
    ok = (end >= 10) & (ref_n > 1e-2)
    if ok.any():
        ratio = (pol_d[ok] * ref_d[ok]).sum(axis=-1) / np.maximum(ref_n[ok] ** 2, 1e-9)
        speed_ratio = float(np.clip(ratio, -1.0, 3.0).mean())
    else:
        speed_ratio = float("nan")

    return dict(
        surv=surv,
        eplen_frac=eplen_frac,
        joint_rms_mean=float(per_joint.mean()),
        joint_rms_worst=float(worst.mean()),
        joint_rms_max=float(per_joint.max()),
        per_joint_rms=per_joint,
        bad_death_frac=_classify_bad_deaths(data, thresholds, horizon, traj_len),
        act_rate=act_rate,
        height_rms=height_rms,
        speed_ratio=speed_ratio,
    )


def evaluate_agent(env, agent_conf, agent_state, *, n_envs=64, n_steps=300,
                   deterministic=True, seed=0, worst_k=3):
    """Roll out the trained policy in `env` and return a list of scalar-diagnostic dicts, one
    per training seed (so an HPO objective can take a seed mean + std). Reuses debug_agent's
    `rollout`; when the agent was trained with `n_seeds > 1` (the Optuna setup uses 2), each
    seed's params are evaluated independently by slicing the leading seed axis and rolling out
    with an `n_seeds == 1` view (otherwise `rollout` would only ever score seed 0).

    Robust to a seed whose policy produced NaN params/rollout: that seed's dict is returned
    with NaN scalars so the caller can penalize it (matches the sweep's "NaN seed -> 9999")."""
    import jax
    from omegaconf import OmegaConf

    thresholds = read_terminal_thresholds(env)
    horizon = int(getattr(env.info, "horizon", n_steps))
    traj_len = int(env.th.len_trajectory(0))

    n_seeds = int(OmegaConf.select(agent_conf.config, "experiment.n_seeds") or 1)

    # build an n_seeds==1 view of agent_conf so rollout() does not collapse to seed 0 itself
    eval_conf = agent_conf.replace(config=agent_conf.config.copy()) \
        if hasattr(agent_conf, "replace") else agent_conf
    OmegaConf.set_struct(eval_conf.config, False)
    eval_conf.config.experiment.n_seeds = 1

    results = []
    for s in range(max(n_seeds, 1)):
        if n_seeds > 1:
            seed_state = jax.tree.map(lambda x: x[s], agent_state)
        else:
            seed_state = agent_state
        try:
            data = rollout(env, eval_conf, seed_state, n_envs, n_steps,
                           deterministic=deterministic, seed=seed)
            if not np.isfinite(data["qpos"]).all():
                raise FloatingPointError("non-finite rollout")
            results.append(_scalars_for_one(data, thresholds, horizon, traj_len, worst_k))
        except Exception as e:                                       # NaN params / blow-up
            print(f"  [agent_diagnostics] seed {s} rollout failed: {type(e).__name__}: {e}")
            results.append(dict(surv=float("nan"), eplen_frac=float("nan"),
                                joint_rms_mean=float("nan"), joint_rms_worst=float("nan"),
                                joint_rms_max=float("nan"), per_joint_rms=np.full(12, np.nan),
                                bad_death_frac=float("nan"), act_rate=float("nan"),
                                height_rms=float("nan"), speed_ratio=float("nan")))
    return results


# ------------------------------------------------------------------------------
# Composite "Metric for Sweep" built from the diagnostics above
# ------------------------------------------------------------------------------
# Baseline = the current best agent (2026-06-10/13-24-37, first trained on the symmetrized
# reference), measured with this module. Tracking terms are normalized to it so each reads
# ~1.0 at baseline; survival/instability are penalty terms that sit at ~0 for the (healthy)
# baseline and only fire when a reward config destabilizes the policy. FILL from
# `python agent_diagnostics.py --path .../13-24-37/PPOJax_saved.pkl` (printed BASELINE block).
BASELINE = dict(
    joint_rms_mean=8.31,    # measured on 2026-06-10/13-24-37 (64 envs x 300 steps, deterministic)
    joint_rms_worst=13.09,  # mean of worst-3 joints: fl_ankle 16.7, br_knee 11.9, fl_knee 10.7
    act_rate=0.0235,        # mean |Δaction| per control step over alive steps
    height_rms=0.0711,      # walks crouched: ~0.33 m held vs the reference's 0.40 m
    speed_err=0.4886,       # |1 - speed_ratio|: HALF the reference pace (~2.0 m vs ~4.1 m in 3 s;
                            # per-env ratio is tight, 0.46-0.54 -> a real pace deficit, not noise)
)
# For reference, the baseline agent at these BASELINE values scores composite ≈ 6.03
# (= W_JMEAN + W_JWORST + W_SMOOTH + W_HEIGHT + W_SPEED + LAM_SURV·(1−0.969)), bad_death = 0.

# Weights, set to favour correcting the CURRENT agent's measured disadvantages (debug_agent on
# 13-24-37): a few badly-tracked joints (fl_ankle/knees) -> joint_rms_worst highest; a held
# ~7 cm crouch -> height term; an ~18% pace deficit -> speed term. The mean-over-joints
# euclidean the sweep used originally was blind to ALL of these. Survival/instability are
# guardrails so the search can't trade tracking for a policy that falls or blows up.
W_JMEAN  = 1.0    # mean per-joint tracking
W_JWORST = 2.0    # WORST joints (the lever the debug plots point at)
W_SMOOTH = 0.5    # action smoothness (anti-jitter; not currently a problem, keep low)
W_HEIGHT = 1.5    # base-height tracking (the crouched-walk disadvantage)
W_SPEED  = 1.0    # forward pace vs reference (the slow-gait disadvantage)
LAM_SURV = 1.0    # survival penalty:   max(0, 1 - surv)        -> falls short of full horizon
LAM_DEATH = 2.0   # instability penalty: bad_death_frac          -> falls / tips / NaN blow-up


def composite_from_scalars(m, baseline=None):
    """Map one seed's scalar dict to the minimizable composite. NaN-in -> 9999 (penalize an
    unstable seed), matching the Optuna objective's convention. Baseline ~= W_JMEAN + W_JWORST
    + W_SMOOTH + W_HEIGHT + W_SPEED (the penalty terms are ~0 for a healthy agent), so a value
    below ~6.03 is better than the current agent."""
    b = baseline or BASELINE
    vals = [m["joint_rms_mean"], m["joint_rms_worst"], m["act_rate"], m["surv"],
            m["bad_death_frac"], m["height_rms"], m["speed_ratio"]]
    if any(v != v for v in vals):                                   # NaN guard
        return 9999.0
    return (W_JMEAN  * m["joint_rms_mean"]  / b["joint_rms_mean"]
            + W_JWORST * m["joint_rms_worst"] / b["joint_rms_worst"]
            + W_SMOOTH * m["act_rate"]        / b["act_rate"]
            + W_HEIGHT * m["height_rms"]      / b["height_rms"]
            + W_SPEED  * abs(1.0 - m["speed_ratio"]) / b["speed_err"]
            + LAM_SURV  * max(0.0, 1.0 - m["surv"])
            + LAM_DEATH * m["bad_death_frac"])


# ------------------------------------------------------------------------------
# CLI: print a baseline block for a saved agent (used to fill BASELINE above)
# ------------------------------------------------------------------------------
def _load_and_eval(path, n_envs, n_steps, worst_k):
    from pathlib import Path
    from omegaconf import OmegaConf
    from loco_mujoco import TaskFactory
    from loco_mujoco.algorithms import PPOJax
    from loco_mujoco.trajectory.dataclasses import Trajectory
    from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf
    from simulation.config import paths

    agent_conf, agent_state = PPOJax.load_agent(str(path))
    config = agent_conf.config
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    traj = Trajectory.load(paths.TRAJ_ADAPTED)
    custom_conf = CustomDatasetConf(traj=traj)
    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = True
    env = factory.make(**config.experiment.env_params, custom_dataset_conf=custom_conf)
    return evaluate_agent(env, agent_conf, agent_state, n_envs=n_envs, n_steps=n_steps,
                          worst_k=worst_k)


def main():
    import argparse
    from simulation.analysis.debug_agent import find_newest_agent
    from simulation.config import paths
    ap = argparse.ArgumentParser(description="Scalar diagnostics + composite for a trained agent.")
    ap.add_argument("--path", type=str, default=None)
    ap.add_argument("--n_envs", type=int, default=64)
    ap.add_argument("--n_steps", type=int, default=300)
    ap.add_argument("--worst_k", type=int, default=3)
    args = ap.parse_args()

    from pathlib import Path
    path = Path(args.path) if args.path else \
        find_newest_agent(paths.TRAINED_AGENTS)
    print(f"Loading agent: {path}")
    per_seed = _load_and_eval(path, args.n_envs, args.n_steps, args.worst_k)

    # aggregate over seeds (single-seed agents -> one entry)
    keys = ["surv", "eplen_frac", "joint_rms_mean", "joint_rms_worst", "joint_rms_max",
            "bad_death_frac", "act_rate", "height_rms", "speed_ratio"]
    agg = {k: float(np.nanmean([m[k] for m in per_seed])) for k in keys}
    pj = np.nanmean(np.stack([m["per_joint_rms"] for m in per_seed]), axis=0)

    print("\n================ SCALAR DIAGNOSTICS ================")
    for k in keys:
        print(f"  {k:16s}: {agg[k]:.4f}")
    print("  per-joint RMS [deg] (worst first):")
    for j in np.argsort(pj)[::-1]:
        print(f"      {JOINT_NAMES[j]:10s} {pj[j]:6.2f}")
    print("\n---- paste into BASELINE in agent_diagnostics.py ----")
    print(f"    joint_rms_mean={agg['joint_rms_mean']:.2f},")
    print(f"    joint_rms_worst={agg['joint_rms_worst']:.2f},")
    print(f"    act_rate={agg['act_rate']:.4f},")
    print(f"    height_rms={agg['height_rms']:.4f},")
    print(f"    speed_err={abs(1.0 - agg['speed_ratio']):.4f},")
    comp = float(np.mean([composite_from_scalars(m) for m in per_seed]))
    print(f"\n  composite (this agent, current BASELINE) = {comp:.3f}")
    print("===================================================")


if __name__ == "__main__":
    main()
