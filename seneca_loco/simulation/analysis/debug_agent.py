"""
debug_agent.py

Rolls out a trained PPO policy in the MJX SenecaBot environment and produces a
suite of diagnostic graphs to help understand *why* the imitation (DeepMimic)
agent misbehaves. Everything is compared against the reference trajectory the
agent is supposed to mimic, so you can see at a glance which joints drift, when
the robot falls, whether actions are saturating, and how the reward decomposes.

Saved figures (in artifacts/figures/agent_debug/):
    1. joint_tracking.png      - actual vs reference angle, per joint, with limit bands
    2. joint_error_rms.png     - RMS tracking error per joint (which joints are worst)
    3. root_tracking.png       - base position / height / orientation vs reference
    4. survival_reward.png     - alive fraction + reward over time, termination histogram
    5. actions.png             - per-joint action traces + saturation fraction
    6. reward_breakdown.png    - weighted reward terms over time + mean contribution
    7. termination.png         - *why* robots die: cause breakdown (instability/NaN vs
                                 height vs orientation vs timeout), height & orientation
                                 traces vs their bounds, the |qvel| instability signature,
                                 a peri-mortem (death-aligned) overlay of what moves before
                                 death, and when-they-die split by cause.

The termination analysis classifies each death by the env's own logic. An episode ends when
the terminal handler flags the state `absorbing` (the ImitationFactory default
`RootPoseTrajTerminalStateHandler`: root height out of [traj_min - margin, traj_max + margin]
OR orientation more than (max_traj_deviation + margin) from the reference), when the horizon
is reached, OR when the observation goes NaN (the env's `done |= isnan(obs)` guard — the
physics blew up). The recorded `absorbing` flag is the authoritative discriminator, so the
plot reports the real cause rather than re-deriving it; thresholds are read off the env.
NB: for the current SenecaBot agent the dominant cause is NaN/instability, NOT falling or
tipping — the robots numerically explode well inside the height/orientation bounds.

Usage:
    python debug_agent.py                      # newest agent under trained_agents/
    python debug_agent.py --path path/to/PPOJax_saved.pkl
    python debug_agent.py --n_envs 64 --n_steps 400 --stochastic
    python debug_agent.py --video              # also record a MuJoCo clip of a fall
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm


def _use_times_new_roman():
    """Render all debug figures in Times New Roman. The msttcorefonts TTFs ship on this box
    but matplotlib's cache may not have indexed them, so register any found explicitly, then
    fall back to metric-compatible serifs (Nimbus Roman / Liberation Serif) if absent."""
    for p in ("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/timesbi.ttf"):
        if Path(p).exists():
            try:
                _fm.fontManager.addfont(p)
            except Exception:
                pass
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",   # serif math to match Times
        "pdf.fonttype": 42,           # embed real glyphs (vector, editable) instead of bitmaps
    })


_use_times_new_roman()

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.trajectory.dataclasses import Trajectory
from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf
from omegaconf import OmegaConf

from simulation.config import paths

# ------------------------------------------------------------------------------
# Layout constants (SenecaBot: 7-DoF free root + 12 hinge joints)
# ------------------------------------------------------------------------------
JOINT_NAMES = [
    "fr_hip", "fr_knee", "fr_ankle",
    "fl_hip", "fl_knee", "fl_ankle",
    "br_hip", "br_knee", "br_ankle",
    "bl_hip", "bl_knee", "bl_ankle",
]
QPOS_JOINT_SLICE = slice(7, 19)   # 12 joint angles
QVEL_JOINT_SLICE = slice(6, 18)   # 12 joint velocities
ROOT_POS_SLICE = slice(0, 3)      # x, y, z of base
ROOT_QUAT_SLICE = slice(3, 7)     # w, x, y, z of base

OUTPUT_DIR = paths.AGENT_DEBUG


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def find_newest_agent(trained_dir: Path) -> Path:
    pkls = list(trained_dir.rglob("PPOJax_saved.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No PPOJax_saved.pkl found under {trained_dir}")
    return max(pkls, key=lambda p: p.stat().st_mtime)


def quat_to_euler(q):
    """(...,4) w,x,y,z -> roll,pitch,yaw in radians."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def first_done_index(done):
    """done: (T, n_envs) bool -> per-env first True index, or T if never."""
    T = done.shape[0]
    idx = np.where(done.any(axis=0), done.argmax(axis=0), T)
    return idx


def root_angular_distance(quat_wxyz, centroid_xyzw):
    """Geodesic angular distance (rad) of root orientation from the reference centroid,
    replicating RootPoseTrajTerminalStateHandler exactly: 2*arccos(clip(dot, -1, 1)) with
    unit, scalar-last quaternions. quat_wxyz: (...,4) scalar-first (MuJoCo qpos[3:7])."""
    q = quat_wxyz[..., [1, 2, 3, 0]]                       # w,x,y,z -> x,y,z,w
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    c = np.asarray(centroid_xyzw, dtype=float)
    c = c / np.maximum(np.linalg.norm(c), 1e-12)
    dot = np.clip(np.einsum("...j,j->...", q, c), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def read_terminal_thresholds(env) -> dict:
    """Pull the active terminal-condition thresholds off the constructed env so the
    termination plot reflects reality rather than hardcoded numbers. Returns a dict with
    height_range, rot_centroid (scalar-last), rot_threshold (rad), and the handler name.
    Falls back to SenecaBot's HeightBased range if a handler without these attrs is used."""
    tsh = getattr(env, "_terminal_state_handler", None)
    name = type(tsh).__name__ if tsh is not None else "unknown"
    hr = getattr(tsh, "root_height_range", None)
    centroid = getattr(tsh, "_centroid_quat", None)
    thr = getattr(tsh, "_valid_threshold", None)
    if hr is None:
        hr = (0.35, 0.7)                                  # SenecaBot HeightBased fallback
    return dict(
        height_range=(float(hr[0]), float(hr[1])),
        rot_centroid=None if centroid is None else np.asarray(centroid, dtype=float),
        rot_threshold=None if thr is None else float(thr),
        handler=name,
    )


# weighted reward terms, in stacking order, with display labels/colors
REWARD_TERMS = [
    ("c_qpos",     "qpos (joint pos)",   "tab:blue"),
    ("c_qvel",     "qvel (joint vel)",   "tab:cyan"),
    ("c_rpos",     "rpos (site pos)",    "tab:green"),
    ("c_rquat",    "rquat (site orient)", "tab:olive"),
    ("c_rvel_rot", "rvel rot",           "tab:purple"),
    ("c_rvel_lin", "rvel lin",           "tab:pink"),
    ("c_rootvel",  "rootvel (root x/y/yaw vel)", "tab:brown"),
    ("c_contact",  "contact pattern",    "tab:red"),
]


def make_reward_component_fn(env, act_low, act_high):
    """Build a single-env function that reproduces MimicReward.__call__ term by
    term and returns each *weighted* contribution to the total reward, so the
    breakdown sums (minus penalties) back to the env reward. Meant to be vmapped."""
    from loco_mujoco.core.utils.math import (calculate_relative_site_quatities,
                                             quaternion_angular_distance,
                                             quat_scalarfirst2scalarlast)
    from loco_mujoco.core.reward.utils import out_of_bounds_action_cost
    from jax._src.scipy.spatial.transform import Rotation as jnp_R
    rf = env._reward_function

    qpos_ind = np.asarray(rf._qpos_ind)
    qvel_ind = np.asarray(rf._qvel_ind)
    quat_mask = np.asarray(rf._quat_in_qpos)          # numpy → static boolean index
    qpos_pos_mask = np.asarray(rf._qpos_pos_mask)     # non-quat positions, honoring track_root_xy
    free_mask = np.asarray(rf._free_joint_qvel_mask)
    free_qpos_ind = np.asarray(rf._free_joint_qpos_ind)   # root free-joint qpos slice
    free_qvel_ind = np.asarray(rf._free_joint_qvel_ind)   # root free-joint qvel slice
    rel_site_ids = np.asarray(rf._rel_site_ids)
    rel_body_ids = np.asarray(rf._rel_body_ids)
    body_rootid = np.asarray(env._model.body_rootid)
    multi_site = len(rel_site_ids) > 1
    dt = float(env.dt)

    # exponents and sum-weights, exactly as configured
    qpe, qve, rpe, rqe, rve = (rf._qpos_w_exp, rf._qvel_w_exp, rf._rpos_w_exp,
                               rf._rquat_w_exp, rf._rvel_w_exp)
    qps, qvs, rps, rqs, rvs = (rf._qpos_w_sum, rf._qvel_w_sum, rf._rpos_w_sum,
                               rf._rquat_w_sum, rf._rvel_w_sum)
    rvroot_s, rvroot_e = rf._rootvel_w_sum, rf._rootvel_w_exp   # root x/y/yaw-vel term
    aob, jac, jtc, arc = (rf._action_out_of_bounds_coeff, rf._joint_acc_coeff,
                          rf._joint_torque_coeff, rf._action_rate_coeff)
    # contact-pattern reward + once-applied smoothness penalties (SENECA local change, 2026-09-27)
    ctw, arw, ajw = rf._contact_w_sum, rf._action_rate_w, rf._action_jerk_w
    if ctw > 0.0:
        contact_site_ids = np.asarray(rf._contact_site_ids)
        ref_contact = jnp.asarray(rf._ref_contact)
        ref_split = jnp.asarray(rf._ref_split_points)

    def _local_root_vel(d):
        # root linear x,y velocity rotated into the base frame, plus yaw rate — exactly
        # as MimicReward's rootvel term (SENECA local change in trajectory_based.py).
        v = jnp.squeeze(d.qvel[free_qvel_ind])
        quat = quat_scalarfirst2scalarlast(jnp.squeeze(d.qpos[free_qpos_ind])[3:7])
        lin_local = jnp_R.from_quat(quat).as_matrix().T @ v[:3]
        return jnp.concatenate([lin_local[:2], jnp.atleast_1d(v[5])])

    def single(data, ref, action, last_qvel, last_action, last_last_action, traj_no, subtraj_step_no):
        qpos, qvel = data.qpos[qpos_ind], data.qvel[qvel_ind]
        qpos_t, qvel_t = ref.qpos[qpos_ind], ref.qvel[qvel_ind]
        qpos_quat = qpos[quat_mask].reshape(-1, 4)
        qpos_quat_t = qpos_t[quat_mask].reshape(-1, 4)

        # joint-position distance uses qpos_pos_mask (excludes root x/y when track_root_xy
        # is False), NOT ~quat_mask — otherwise the metres-large forward-walk root-x error
        # saturates exp(-qpe·dist) to ~0 and the qpos term reads as a near-zero artifact.
        qpos_dist = jnp.mean(jnp.square(qpos[qpos_pos_mask] - qpos_t[qpos_pos_mask]))
        qpos_dist += jnp.mean(quaternion_angular_distance(qpos_quat, qpos_quat_t, jnp))
        qvel_dist = jnp.mean(jnp.square(qvel - qvel_t))
        qpos_r = jnp.exp(-qpe * qpos_dist)
        qvel_r = jnp.exp(-qve * qvel_dist)

        # root x/y/yaw-velocity tracking term (SENECA local change)
        rootvel_dist = jnp.mean(jnp.square(_local_root_vel(data) - _local_root_vel(ref)))
        rootvel_r = jnp.nan_to_num(jnp.exp(-rvroot_e * rootvel_dist), nan=0.0)

        if multi_site:
            rp, ra, rv = calculate_relative_site_quatities(
                data, rel_site_ids, rel_body_ids, body_rootid, jnp)
            rpt, rat, rvt = calculate_relative_site_quatities(
                ref, rel_site_ids, rel_body_ids, body_rootid, jnp)
            rpos_r = jnp.exp(-rpe * jnp.mean(jnp.square(rp - rpt)))
            rquat_r = jnp.exp(-rqe * jnp.mean(jnp.square(ra - rat)))
            rvr_r = jnp.exp(-rve * jnp.mean(jnp.square(rv[:, :3] - rvt[:, :3])))
            rvl_r = jnp.exp(-rve * jnp.mean(jnp.square(rv[:, 3:] - rvt[:, 3:])))
        else:
            rpos_r = rquat_r = rvr_r = rvl_r = 0.0

        # penalties (replicated exactly, incl. the coeff-squared weighting)
        oob = -out_of_bounds_action_cost(action, act_low, act_high, jnp)
        acc_norm = jnp.sum(jnp.square(data.qvel[~free_mask] - last_qvel[~free_mask]) / dt)
        tor_norm = jnp.sum(jnp.square(data.qfrc_actuator[~free_mask]))
        arate_norm = jnp.sum(jnp.square(action - last_action))
        penalties = (aob * oob + jac * (jac * -acc_norm)
                     + jtc * (jtc * -tor_norm) + arc * (arc * -arate_norm))
        penalties = jnp.maximum(penalties, -1.0)

        return dict(
            c_qpos=qps * qpos_r, c_qvel=qvs * qvel_r,
            c_rpos=rps * rpos_r, c_rquat=rqs * rquat_r,
            c_rvel_rot=rvs * rvr_r, c_rvel_lin=rvs * rvl_r,
            c_rootvel=rvroot_s * rootvel_r,
            c_penalty=penalties,
            c_contact=(ctw * jnp.mean((data.site_xpos[contact_site_ids, 2] <= rf._contact_height)
                                      == ref_contact[ref_split[traj_no] + subtraj_step_no])
                       if ctw > 0.0 else jnp.zeros(())),
            c_smooth=-(arw * jnp.sum(jnp.square(action - last_action))
                       + ajw * jnp.sum(jnp.square(action - 2.0 * last_action + last_last_action))),
        )

    return single


# ------------------------------------------------------------------------------
# Rollout
# ------------------------------------------------------------------------------
FOOT_GEOM_NAMES = ("fr_foot", "fl_foot", "br_foot", "bl_foot")


def rollout(env, agent_conf, agent_state, n_envs, n_steps, deterministic, seed,
            collect_dynamics=False):
    """Parallel MJX rollout. Returns a dict of numpy arrays shaped (T, n_envs, ...).

    With ``collect_dynamics=True`` two extra fields are added (default off keeps the
    output byte-identical for existing callers):
        qfrc_actuator  (T, n_envs, nv)  actuator generalized forces — for cost of transport
        foot_contact   (T, n_envs, 4)   per-foot floor contact flag (fr, fl, br, bl) — for the
                                         gait/phase diagram
    """
    network = agent_conf.network
    ts = agent_state.train_state
    cfg = agent_conf.config.experiment
    if cfg.n_seeds > 1:                       # collapse seed dim if present
        ts = jax.tree.map(lambda x: x[0], ts)
    params, run_stats = ts.params, ts.run_stats

    if collect_dynamics:
        import mujoco
        _model = env.get_model()
        _foot_ids = jnp.array([mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_GEOM, g)
                               for g in FOOT_GEOM_NAMES])

    def policy(obs):
        (pi, _), _ = network.apply({"params": params, "run_stats": run_stats},
                                   obs, mutable=["run_stats"])
        return pi.mean() if deterministic else pi.sample(seed=jax.random.key(0))

    th_data = env.th.traj.data
    act_low = np.asarray(env.info.action_space.low)
    act_high = np.asarray(env.info.action_space.high)
    act_dim = act_low.shape[0]
    # the per-term breakdown reproduces MimicReward; agents trained with another reward (e.g. the
    # SlopeLocomotionReward slope fine-tune) get the rollout without it
    has_mimic = type(env._reward_function).__name__ == "MimicReward"
    component_fn = make_reward_component_fn(env, act_low, act_high) if has_mimic else None

    def step(carry, _):
        state, rng, last_action, last_last_action = carry
        rng, k = jax.random.split(rng)
        obs = state.observation
        action = policy(obs)
        if not deterministic:
            # re-sample with fresh key per step
            (pi, _), _ = network.apply({"params": params, "run_stats": run_stats},
                                       obs, mutable=["run_stats"])
            action = pi.sample(seed=k)
        new_state = jax.vmap(env.mjx_step)(state, action)

        ts_ = new_state.additional_carry.traj_state
        ref = jax.vmap(lambda tn, sn: th_data.get(tn, sn, jnp))(
            ts_.traj_no, ts_.subtraj_step_no)

        # reward breakdown: last_qvel = qvel before this step (matches MimicReward)
        comps = jax.vmap(component_fn)(new_state.data, ref, action,
                                       state.data.qvel, last_action, last_last_action,
                                       ts_.traj_no, ts_.subtraj_step_no) if has_mimic else {}

        out = dict(
            qpos=new_state.data.qpos,
            qvel=new_state.data.qvel,
            ref_qpos=ref.qpos,
            ref_qvel=ref.qvel,
            action=action,
            reward=new_state.reward,
            done=new_state.done,
            absorbing=new_state.absorbing,
            # subtraj phase: lets us tell a trajectory-end/wrap truncation (phase near the
            # clip end at the terminating step) apart from a true sim blow-up.
            subtraj_step_no=ts_.subtraj_step_no,
            **comps,
        )
        if collect_dynamics:
            out["qfrc_actuator"] = new_state.data.qfrc_actuator
            # MJX hides the contact array in this version, so detect stance geometrically:
            # collect each foot geom's world-z; the caller thresholds it against the foot-sphere
            # radius (foot in contact when its lowest point sits on the floor).
            out["foot_z"] = new_state.data.geom_xpos[:, _foot_ids, 2]    # (n_envs, 4)
        # the env re-inits the reward state (last actions = 0) on reset; mirror that
        reset = new_state.done.astype(bool)[:, None]
        return (new_state, rng, jnp.where(reset, 0.0, action),
                jnp.where(reset, 0.0, last_action)), out

    rng = jax.random.key(seed)
    rng, kreset = jax.random.split(rng)
    init_state = jax.vmap(env.mjx_reset)(jax.random.split(kreset, n_envs))
    last_action0 = jnp.zeros((n_envs, act_dim))
    (_, _, _, _), traj = jax.lax.scan(step, (init_state, rng, last_action0, last_action0),
                                   None, length=n_steps)
    return jax.tree.map(lambda x: np.asarray(x), traj)


# ------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------
def _alive_mean(x, alive):
    """Mean over envs, weighted by alive mask. x:(T,n_envs,..), alive:(T,n_envs).
    Returns NaN for timesteps where no env is alive, so plots show a gap instead of a
    misleading flat 0 once the whole population has terminated (the dead-population tail)."""
    a = alive[..., None] if x.ndim == 3 else alive
    cnt = a.sum(axis=1)
    s = (x * a).sum(axis=1)
    return np.where(cnt < 0.5, np.nan, s / np.maximum(cnt, 1e-6))


def _unwrap_clip_wrap(x, min_drop=1.0):
    """Remove the large negative jumps a *looping* reference clip makes when its world position
    snaps from clip-end back to clip-start (e.g. base-x dropping ~85 m -> ~0.65 m). x:(T, n_envs).
    Adds each such drop back so every column is a continuous cumulative path; forward walking
    never moves >``min_drop`` m backward in one control step, so genuine motion is untouched.
    Done PER ENV before any cross-env averaging — averaging the raw wrapping signal across
    RSI-randomised phases is what produced the sawtooth artifact in the base-x panel."""
    x = np.asarray(x, dtype=float)
    d = np.diff(x, axis=0)
    add = np.where(d < -min_drop, -d, 0.0)
    corr = np.zeros_like(x)
    corr[1:] = np.cumsum(add, axis=0)
    return x + corr


def _gait_phase_average(pol, ref, n_phase=100, ref_col=0):
    """Fold the per-joint time series into ONE representative gait cycle. Cycle boundaries are
    the troughs of the *reference* FR-hip angle (col 0) — the same boundary the augmentation
    pipeline uses — and policy & reference share the env's phase clock, so both are segmented by
    the same troughs. Each complete cycle is resampled to ``n_phase`` points (% of gait) and
    averaged across cycles.

    pol, ref: (T, 12) mean-over-envs joint angles [deg]. Returns
    (phase_pct(n_phase,), pol_mean(n_phase,12), ref_mean(n_phase,12), pol_std(n_phase,12)) or
    None if fewer than two clean cycles are detectable (caller falls back to the time axis)."""
    from scipy.signal import find_peaks
    sig = ref[:, ref_col]
    finite = np.isfinite(sig)
    if finite.sum() < 10:
        return None
    filled = np.where(finite, sig, np.nanmax(sig[finite]))      # fill gaps high so they aren't troughs
    prom = max(float(np.nanstd(sig[finite])) * 0.4, 0.5)
    troughs, _ = find_peaks(-filled, prominence=prom, distance=15)
    troughs = [int(i) for i in troughs if finite[i]]
    if len(troughs) < 2:
        return None
    xp = np.linspace(0.0, 1.0, n_phase)
    cyc_p, cyc_r = [], []
    for b0, b1 in zip(troughs[:-1], troughs[1:]):
        sp, sr = pol[b0:b1], ref[b0:b1]
        if b1 - b0 < 5 or not np.isfinite(sp).all() or not np.isfinite(sr).all():
            continue
        src = np.linspace(0.0, 1.0, b1 - b0)
        cyc_p.append(np.stack([np.interp(xp, src, sp[:, k]) for k in range(sp.shape[1])], axis=1))
        cyc_r.append(np.stack([np.interp(xp, src, sr[:, k]) for k in range(sr.shape[1])], axis=1))
    if not cyc_p:
        return None
    P, R = np.stack(cyc_p), np.stack(cyc_r)                     # (n_cycles, n_phase, 12)
    return xp * 100.0, P.mean(0), R.mean(0), P.std(0)


def plot_joint_tracking(data, alive, t, ranges, out, x_mode="gait"):
    """Policy vs reference joint angles. With ``x_mode="gait"`` the rollout is folded into one
    representative gait cycle (x = % of gait cycle) and each panel's y-axis is auto-scaled to
    its own data range so small tracking differences are legible (no full joint-limit band)."""
    qj = np.rad2deg(data["qpos"][:, :, QPOS_JOINT_SLICE])
    rj = np.rad2deg(data["ref_qpos"][:, :, QPOS_JOINT_SLICE])
    qj_m, rj_m = _alive_mean(qj, alive), _alive_mean(rj, alive)

    pa = _gait_phase_average(qj_m, rj_m) if x_mode == "gait" else None
    if pa is not None:
        x, pol, ref, pol_sd = pa
        xlabel, suptitle = "% gait cycle", \
            "Joint tracking — policy (blue) vs reference (orange), averaged over the gait cycle"
    else:
        x, pol, ref, pol_sd = t, qj_m, rj_m, None
        xlabel, suptitle = "time [s]", "Joint tracking — policy (blue) vs reference (orange)"

    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
    fig.suptitle(suptitle, fontsize=14)
    for k, ax in enumerate(axes.flat):
        ax.plot(x, ref[:, k], color="tab:orange", lw=1.8, label="reference")
        ax.plot(x, pol[:, k], color="tab:blue", lw=1.5, label="policy")
        if pol_sd is not None:
            ax.fill_between(x, pol[:, k] - pol_sd[:, k], pol[:, k] + pol_sd[:, k],
                            color="tab:blue", alpha=0.18, lw=0)
        # y-axis: tight to the data (policy band + reference) so differences are visible
        lo = np.nanmin([np.nanmin(ref[:, k]),
                        np.nanmin(pol[:, k] - (0 if pol_sd is None else pol_sd[:, k]))])
        hi = np.nanmax([np.nanmax(ref[:, k]),
                        np.nanmax(pol[:, k] + (0 if pol_sd is None else pol_sd[:, k]))])
        pad = 0.12 * (hi - lo) + 0.5
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(JOINT_NAMES[k]); ax.grid(True, alpha=0.3)
        if k % 3 == 0: ax.set_ylabel("angle [deg]")
        if k >= 9: ax.set_xlabel(xlabel)
    # single shared legend, centered just below the title (clearer than a per-panel legend)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out, dpi=130); plt.close(fig)


def plot_joint_error(data, alive, out):
    err = np.rad2deg(data["qpos"][:, :, QPOS_JOINT_SLICE]
                     - data["ref_qpos"][:, :, QPOS_JOINT_SLICE])
    a = alive[..., None]
    rms = np.sqrt((err ** 2 * a).sum(axis=(0, 1)) / np.maximum(a.sum(), 1e-6))
    order = np.argsort(rms)[::-1]
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.RdYlGn_r(rms[order] / max(rms.max(), 1e-6))
    ax.bar(np.array(JOINT_NAMES)[order], rms[order], color=colors)
    ax.set_ylabel("RMS tracking error [deg]")
    ax.set_title("Per-joint RMS tracking error (worst → best)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(rms[order]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_root_tracking(data, alive, t, out):
    pos = data["qpos"][:, :, ROOT_POS_SLICE].astype(float).copy()
    rpos = data["ref_qpos"][:, :, ROOT_POS_SLICE].astype(float).copy()
    eul = np.rad2deg(quat_to_euler(data["qpos"][:, :, ROOT_QUAT_SLICE]))
    reul = np.rad2deg(quat_to_euler(data["ref_qpos"][:, :, ROOT_QUAT_SLICE]))
    eul_m, reul_m = _alive_mean(eul, alive), _alive_mean(reul, alive)
    # z (height) is a bounded, per-env-comparable absolute quantity, so average it directly.
    posz_m, rposz_m = _alive_mean(pos[:, :, 2], alive), _alive_mean(rpos[:, :, 2], alive)

    def _mean_travel(world_xy):
        """Mean per-env DISPLACEMENT FROM START for one world axis (x or y), (T, n_envs)->(T,).
        Each env is referenced to its OWN start and the looping reference clip's end->start jump
        (~85 m) is unwrapped BEFORE averaging. Critically we subtract the per-env start first, so
        every env contributes only its few-metre travel: averaging the *absolute* world position
        (envs begin anywhere in 2..85 m under random-start) is corrupted the instant a high-offset
        env leaves the alive set, which is what produced the early sawtooth in this panel."""
        disp = world_xy - world_xy[0:1, :]      # per-env displacement from its own start
        disp = _unwrap_clip_wrap(disp)          # remove the clip-wrap discontinuity per env
        return _alive_mean(disp, alive)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # World x/y absolute position is NOT a tracked target (track_root_xy=False); only forward/
    # lateral *travel* is comparable, so each panel shows mean per-env displacement from start.
    labels = ["base x — displacement from start [m]\n(world x/y untracked: track_root_xy=False)",
              "base y — displacement from start [m]", "base z (height) [m]"]
    travel = [_mean_travel(pos[:, :, 0]), _mean_travel(pos[:, :, 1]), posz_m]
    rtravel = [_mean_travel(rpos[:, :, 0]), _mean_travel(rpos[:, :, 1]), rposz_m]
    for i in range(3):
        ax = axes[0, i]
        ax.plot(t, rtravel[i], color="tab:orange", lw=1.6, label="reference")
        ax.plot(t, travel[i], color="tab:blue", lw=1.4, label="policy")
        ax.set_title(labels[i], fontsize=12); ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        if i == 0: ax.legend(fontsize=11)
    elabels = ["roll [deg]", "pitch [deg]", "yaw [deg]"]
    for i in range(3):
        ax = axes[1, i]
        ax.plot(t, reul_m[:, i], color="tab:orange", lw=1.6)
        ax.plot(t, eul_m[:, i], color="tab:blue", lw=1.4)
        ax.set_title(elabels[i], fontsize=13); ax.set_xlabel("time [s]", fontsize=12)
        ax.tick_params(labelsize=11); ax.grid(True, alpha=0.3)
    fig.suptitle("Base/root tracking — policy (blue) vs reference (orange)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=130); plt.close(fig)


def plot_survival_reward(data, alive, t, dt, out):
    alive_frac = alive.mean(axis=1)
    rew = data["reward"]
    rew_m = _alive_mean(rew, alive)
    rew_std = np.sqrt(_alive_mean((rew - rew_m[:, None]) ** 2, alive))
    fd = first_done_index(data["done"]) * dt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(t, alive_frac, color="tab:green")
    axes[0].fill_between(t, 0, alive_frac, color="tab:green", alpha=0.2)
    axes[0].set_title("Fraction of robots still alive")
    axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("alive fraction")
    axes[0].set_ylim(0, 1.02); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, rew_m, color="tab:blue")
    axes[1].fill_between(t, rew_m - rew_std, rew_m + rew_std, color="tab:blue", alpha=0.2)
    axes[1].set_title("Per-step reward (alive envs, mean ± std)")
    axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("reward"); axes[1].grid(True, alpha=0.3)

    survivors = (data["done"].any(axis=0) == False).sum()
    axes[2].hist(fd[data["done"].any(axis=0)], bins=20, color="tab:red", alpha=0.8)
    axes[2].set_title(f"When robots fall  ({survivors}/{data['done'].shape[1]} survive full episode)")
    axes[2].set_xlabel("time of termination [s]"); axes[2].set_ylabel("count")
    axes[2].grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_reward_breakdown(data, alive, t, out):
    """Stacked weighted reward terms over time + time-averaged contribution, so
    you can see exactly which term holds the total up despite poor tracking."""
    # exclude each env's terminating step: the env resets state on `done`, so the
    # recorded data there is post-reset and no longer matches the reward it logged
    strict = alive * (~data["done"].astype(bool))
    term_m = {k: _alive_mean(data[k], strict) for k, _, _ in REWARD_TERMS}
    pen_m = _alive_mean(data["c_penalty"] + data["c_smooth"], strict)
    total_m = _alive_mean(data["reward"], strict)
    # element-wise correctness check vs the env reward (env clips total at >= 0)
    recon_raw = np.clip(np.sum([data[k] for k, _, _ in REWARD_TERMS], axis=0)
                        + data["c_penalty"] + data["c_smooth"], 0.0, None)
    recon_err = np.abs((recon_raw - data["reward"]) * strict).max()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5),
                             gridspec_kw={"width_ratios": [2, 1]})
    labels = [lab for _, lab, _ in REWARD_TERMS]
    colors = [c for _, _, c in REWARD_TERMS]
    axes[0].stackplot(t, *[term_m[k] for k, _, _ in REWARD_TERMS],
                      labels=labels, colors=colors, alpha=0.85)
    axes[0].plot(t, total_m, color="black", lw=1.8, label="total reward")
    axes[0].plot(t, pen_m, color="tab:red", lw=1.0, ls="--", label="penalties")
    axes[0].set_title("Reward decomposition over time (weighted contributions)", fontsize=15)
    axes[0].set_xlabel("time [s]", fontsize=12); axes[0].set_ylabel("reward contribution", fontsize=12)
    axes[0].tick_params(labelsize=11)
    axes[0].legend(fontsize=9, ncol=2, loc="upper right"); axes[0].grid(True, alpha=0.3)

    # time-averaged contribution of each term over alive steps
    a = strict
    avg = [(data[k] * a).sum() / np.maximum(a.sum(), 1e-6) for k, _, _ in REWARD_TERMS]
    avg.append(((data["c_penalty"] + data["c_smooth"]) * a).sum() / np.maximum(a.sum(), 1e-6))
    bar_labels = labels + ["penalties"]
    bar_colors = colors + ["tab:red"]
    order = np.argsort(avg)[::-1]
    axes[1].barh(np.array(bar_labels)[order], np.array(avg)[order],
                 color=np.array(bar_colors)[order])
    axes[1].set_title("Mean contribution per term\n"
                      f"(total ≈ {total_m[strict.any(1)].mean():.3f}, "
                      f"recon err {recon_err:.1e})", fontsize=15)
    axes[1].set_xlabel("mean weighted reward", fontsize=12)
    axes[1].tick_params(labelsize=11); axes[1].grid(True, axis="x", alpha=0.3)
    axes[1].invert_yaxis()
    for i, v in enumerate(np.array(avg)[order]):
        axes[1].text(v, i, f" {v:.3f}", va="center", fontsize=9.5)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_actions(data, alive, t, act_low, act_high, out):
    act = data["action"]
    act_m = _alive_mean(act, alive)
    a = alive[..., None]
    # per-env min/max envelope across alive envs (so a single-env excursion to a bound is
    # visible — the mean alone hides it, which makes the saturation bars look unexplained).
    act_env = np.where(a > 0, act, np.nan)
    act_lo_all = np.nanmin(np.nanmin(act_env, axis=1), axis=1)   # (T,) min over envs & joints
    act_hi_all = np.nanmax(np.nanmax(act_env, axis=1), axis=1)
    # saturation: fraction of (alive) (env,step) samples within 2% of a bound, per joint
    span = (act_high - act_low)
    near = (act <= act_low + 0.02 * span) | (act >= act_high - 0.02 * span)
    sat = (near * a).sum(axis=(0, 1)) / np.maximum(a.sum(), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                             gridspec_kw={"width_ratios": [2, 1]})
    axes[0].fill_between(t, act_lo_all, act_hi_all, color="gray", alpha=0.15,
                         label="per-env min–max (all joints)")
    for k in range(act.shape[-1]):
        axes[0].plot(t, act_m[:, k], lw=1.0, label=JOINT_NAMES[k])
    axes[0].axhline(float(np.mean(act_low)), color="gray", ls="--", lw=0.8)
    axes[0].axhline(float(np.mean(act_high)), color="gray", ls="--", lw=0.8)
    axes[0].set_title("Action per joint — across-env mean (lines) + per-env min–max (gray band)")
    axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("action")
    axes[0].legend(fontsize=7, ncol=2, loc="upper right"); axes[0].grid(True, alpha=0.3)

    # absolute colour scale (NOT normalized to sat.max(), which painted a negligible 0.03%
    # bright red): green→red over 0–10% of samples saturated.
    sat_pct = sat * 100
    colors = plt.cm.RdYlGn_r(np.clip(sat_pct / 10.0, 0, 1))
    axes[1].barh(JOINT_NAMES, sat_pct, color=colors)
    axes[1].set_title("Action saturation\n(% of (env,step) samples within 2% of a bound)")
    axes[1].set_xlabel("% of samples saturated"); axes[1].grid(True, axis="x", alpha=0.3)
    axes[1].set_xlim(0, max(sat_pct.max() * 1.3, 1.0))   # fixed floor so tiny values look tiny
    axes[1].invert_yaxis()
    for i, v in enumerate(sat_pct):
        if v > 0:
            axes[1].text(v, i, f" {v:.2g}%", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


# ------------------------------------------------------------------------------
# Termination ("why they die") analysis
# ------------------------------------------------------------------------------
# Cause categories + display colors, used across every panel for consistency.
# The PRIMARY discriminator is the recorded `absorbing` flag (authoritative): a `done` that
# is NOT absorbing and not at the horizon is the env's NaN-observation guard firing, i.e. the
# physics blew up (instability) — distinct from the robot actually falling/tipping out of the
# RootPoseTraj healthy bounds (absorbing=True).
CAUSE_LABELS = ["instability\n(NaN blow-up)", "height\n(fall/sink)",
                "orientation\n(tip over)", "timeout\n(horizon)", "survived\n(window)",
                "traj end\n(clip ran out)"]
CAUSE_COLORS = ["tab:red", "tab:orange", "tab:purple", "tab:gray", "tab:green", "tab:blue"]
# index: 0 = instability, 1 = height, 2 = orientation, 3 = timeout, 4 = survived,
#        5 = trajectory-end/wrap (the demo clip ran out -> env NaN-obs guard at the
#            sub-trajectory boundary; a benign truncation, NOT a sim blow-up)
DEATH_CAUSES = (0, 1, 2, 5)                                # the categories that are deaths


def _peri_mortem_mean(sig, term_idx, died, window):
    """Death-aligned mean of a (T, n_envs) signal: for each dying env, take the `window`
    steps ending at its last pre-reset step (term_idx) and average across envs that have a
    full window. Returns (window,) or None if no env qualifies."""
    keep = died & (term_idx >= window - 1)
    if not keep.any():
        return None
    cols = []
    for e in np.where(keep)[0]:
        end = term_idx[e]
        cols.append(sig[end - window + 1: end + 1, e])
    return np.mean(np.stack(cols, axis=0), axis=0)


def plot_termination_analysis(data, t, dt, thresholds, horizon, traj_len, out):
    """Answer *why* robots don't finish the episode.

    An episode ends (`mjx_step`) when EITHER the env's terminal handler flags the state
    `absorbing` (RootPoseTraj: root height out of band OR orientation too far from reference),
    OR the horizon is reached, OR the observation goes NaN (the env's `done |= isnan(obs)`
    guard). The recorded `absorbing` flag plus the trajectory phase discriminate the cause:

        done & absorbing                              -> fell (height) / tipped (orientation)
        done & ~absorbing & step >= horizon           -> timeout (reached horizon)
        done & ~absorbing & phase near clip end       -> trajectory-end/wrap truncation
        done & ~absorbing & phase mid-clip            -> instability (NaN / sim blow-up)
        never done in the rollout window              -> survived (window)

    IMPORTANT: the NaN-obs guard fires *intrinsically* at the sub-trajectory boundary — when
    the reference clip runs out and the handler wraps it, the goal observation goes NaN for
    any drifted state, independent of the policy (verified: even a zero-torque policy and a
    perfectly clean cyclic clip NaN at the wrap). That is a benign truncation, NOT a physics
    blow-up, and must not be reported as "instability" — hence the phase-based split using the
    recorded `subtraj_step_no`. (This corrects the earlier diagnosis that read these as NaN
    instability; see CLAUDE.md.)

    Height vs orientation is only sub-classified for absorbing deaths (whichever bound the
    state is closest to). Note on indexing: MJX resets in the same step `done` fires, so the
    qpos recorded *at* the done step is post-reset; we read displayed state from
    `term_idx = death-1` (the last real pre-reset step).
    """
    done = data["done"].astype(bool)
    absb = data["absorbing"].astype(bool)
    T, N = done.shape
    died = done.any(axis=0)
    death_idx = first_done_index(done)                    # T if survived
    di = np.clip(death_idx, 0, T - 1)
    term_idx = np.clip(death_idx - 1, 0, T - 1)           # last pre-reset step
    ar = np.arange(N)

    low, high = thresholds["height_range"]
    span = max(high - low, 1e-6)
    centroid = thresholds["rot_centroid"]
    rot_thr = thresholds["rot_threshold"]
    have_rot = centroid is not None and rot_thr is not None

    # full (T, N) signals
    H = data["qpos"][:, :, 2]                              # root height
    A = (np.rad2deg(root_angular_distance(data["qpos"][:, :, ROOT_QUAT_SLICE], centroid))
         if have_rot else np.zeros_like(H))
    QV = np.linalg.norm(data["qvel"], axis=-1)            # |qvel| — instability signature
    jrms = np.rad2deg(np.sqrt(np.mean(
        (data["qpos"][:, :, QPOS_JOINT_SLICE] - data["ref_qpos"][:, :, QPOS_JOINT_SLICE]) ** 2,
        axis=-1)))

    # ---- cause classification (authoritative: uses the recorded `absorbing` flag) --------
    h_term, a_term = H[term_idx, ar], A[term_idx, ar]
    absorb_at_death = absb[di, ar] & died
    is_timeout = died & (~absorb_at_death) & (death_idx >= horizon - 1)
    nonabsorb_early = died & (~absorb_at_death) & (~is_timeout)
    # trajectory-end / wrap truncation vs genuine instability. A non-absorbing, pre-horizon
    # `done` is the env's NaN-obs guard. It fires intrinsically at the sub-trajectory
    # boundary (the clip runs out and wraps) -- a benign truncation, NOT a sim blow-up. We
    # tell them apart by the trajectory phase at the terminating step: near the clip end
    # (within a few steps of traj_len) => the clip ran out; otherwise => real instability.
    sub_term = (data["subtraj_step_no"][term_idx, ar] if "subtraj_step_no" in data
                else np.zeros(N))
    is_wrap = nonabsorb_early & (sub_term >= max(traj_len - 3, 0))
    is_instab = nonabsorb_early & (~is_wrap)
    # for absorbing deaths, split height vs orientation by closest normalized margin
    m_height = np.minimum((h_term - low) / span, (high - h_term) / span)
    m_rot = ((rot_thr - np.deg2rad(a_term)) / max(rot_thr, 1e-6)
             if have_rot else np.full(N, np.inf))
    cause = np.full(N, 4)                                  # default: survived
    cause[is_instab] = 0
    cause[absorb_at_death] = np.where(m_rot[absorb_at_death] < m_height[absorb_at_death], 2, 1)
    cause[is_timeout] = 3
    cause[is_wrap] = 5
    counts = np.array([(cause == k).sum() for k in range(len(CAUSE_LABELS))])

    d = died & (death_idx < T)
    cdot = np.array(CAUSE_COLORS, dtype=object)[cause]     # per-env color by cause

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    n_inst, n_wrap = counts[0], counts[5]
    fig.suptitle(f"Why robots don't finish the episode  —  handler: {thresholds['handler']}"
                 f"   ({died.sum()}/{N} died,  {n_wrap} clip-end truncations,  "
                 f"{n_inst} true instability)", fontsize=14)

    # (0,0) headline: cause breakdown
    ax = axes[0, 0]
    nc = len(CAUSE_LABELS)
    bars = ax.bar(range(nc), counts, color=CAUSE_COLORS, alpha=0.85)
    ax.set_xticks(range(nc)); ax.set_xticklabels(CAUSE_LABELS, fontsize=7)
    ax.set_ylabel("number of robots"); ax.set_title("Cause of termination")
    ax.grid(True, axis="y", alpha=0.3)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c}\n{100*c/max(N,1):.0f}%",
                ha="center", va="bottom", fontsize=8)

    # (0,1) height traces vs the healthy band (death dots colored by cause)
    ax = axes[0, 1]
    ax.plot(t, H, color="tab:blue", lw=0.5, alpha=0.20)
    ax.axhspan(low, high, color="tab:green", alpha=0.08)
    ax.axhline(low, color="tab:red", ls="--", lw=1.0, label=f"low {low:.2f} m")
    ax.axhline(high, color="tab:orange", ls="--", lw=1.0, label=f"high {high:.2f} m")
    ax.scatter(term_idx[d] * dt, h_term[d], s=16, c=list(cdot[d]), zorder=3,
               edgecolors="k", linewidths=0.3)
    ax.set_title("Root height vs healthy band\n(deaths far inside band ⇒ not a height failure)")
    ax.set_xlabel("time [s]"); ax.set_ylabel("base height [m]")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (0,2) orientation deviation vs its threshold
    ax = axes[0, 2]
    if have_rot:
        ax.plot(t, A, color="tab:purple", lw=0.5, alpha=0.20)
        ax.axhline(np.rad2deg(rot_thr), color="tab:red", ls="--", lw=1.0,
                   label=f"threshold {np.rad2deg(rot_thr):.0f}°")
        ax.scatter(term_idx[d] * dt, a_term[d], s=16, c=list(cdot[d]), zorder=3,
                   edgecolors="k", linewidths=0.3)
        ax.set_title("Root orientation deviation from reference\n(deaths far below ⇒ not a tip-over)")
        ax.set_xlabel("time [s]"); ax.set_ylabel("angular distance [deg]")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no orientation threshold\n(height-only handler)",
                ha="center", va="center", fontsize=11); ax.axis("off")

    # (1,0) |qvel| trajectories — the instability signature (death dots colored by cause)
    ax = axes[1, 0]
    ax.plot(t, QV, color="tab:red", lw=0.5, alpha=0.20)
    ax.scatter(term_idx[d] * dt, QV[term_idx[d], ar[d]], s=16, c=list(cdot[d]), zorder=3,
               edgecolors="k", linewidths=0.3)
    ax.set_title("|qvel| (whole-body speed) — instability signature")
    ax.set_xlabel("time [s]"); ax.set_ylabel("|qvel|  [rad,m /s]")
    ax.grid(True, alpha=0.3)

    # (1,1) peri-mortem: what moves before death (death-aligned, normalized overlay)
    ax = axes[1, 1]
    W = max(int(round(1.0 / dt)), 5)                       # ~1 s window
    rel_t = (np.arange(W) - (W - 1)) * dt
    series = [("height", H, "tab:orange"), ("orient. dev.", A, "tab:purple"),
              ("|qvel|", QV, "tab:red"), ("joint RMS err", jrms, "tab:blue"),
              ("reward", data["reward"], "black")]
    plotted = False
    for name, sig, col in series:
        if name == "orient. dev." and not have_rot:
            continue
        m = _peri_mortem_mean(sig, term_idx, died, W)
        if m is None:
            continue
        rng = m.max() - m.min()
        norm = (m - m.min()) / rng if rng > 1e-9 else np.zeros_like(m)
        ax.plot(rel_t, norm, color=col, lw=1.6, label=name)
        plotted = True
    ax.axvline(0.0, color="gray", ls=":", lw=1.0)
    ax.set_title(f"Peri-mortem (death-aligned, normalized)\nmean over dying envs, last {W*dt:.1f} s")
    ax.set_xlabel("time before death [s]"); ax.set_ylabel("normalized (min–max)")
    if plotted:
        ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # (1,2) when they die, stacked by cause
    ax = axes[1, 2]
    bins = np.linspace(0, T * dt, 21)
    stacks = [term_idx[(cause == k) & died] * dt for k in DEATH_CAUSES]
    ax.hist(stacks, bins=bins, stacked=True, color=[CAUSE_COLORS[k] for k in DEATH_CAUSES],
            label=[CAUSE_LABELS[k].replace("\n", " ") for k in DEATH_CAUSES], alpha=0.85)
    ax.set_title("When robots die, by cause")
    ax.set_xlabel("time of termination [s]"); ax.set_ylabel("count")
    ax.legend(fontsize=7); ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=130); plt.close(fig)
    return cause, counts


def record_fall_video(path, out_path, n_steps, stochastic):
    """Record a fall clip in a *separate process*. This is required: running the MJX/warp GPU
    rollout in-process corrupts the MuJoCo render context that the recorder needs (the viewer
    then builds garbage scene geoms and crashes). A fresh subprocess that only does CPU
    rendering — never importing/Running MJX — sidesteps that entirely. Returns the video path,
    or None. Raises on subprocess failure so the caller can report it."""
    cmd = [sys.executable, os.path.abspath(__file__), "--record_only",
           "--path", str(path), "--video_steps", str(n_steps), "--video_out", str(out_path)]
    if stochastic:
        cmd.append("--stochastic")
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")                    # offscreen GL on headless boxes
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in res.stdout.splitlines():
        if line.startswith("VIDEO_PATH="):
            v = line[len("VIDEO_PATH="):].strip()
            return v if v not in ("", "None") else None
    raise RuntimeError((res.stderr or res.stdout or "no output")[-600:])


def _record_fall_inproc(config, custom_conf, factory, agent_conf, agent_state,
                        out_path, n_steps, deterministic):
    """Record a CPU-MuJoCo clip of the policy up to the *first* fall so the failure mode can be
    watched. We roll out a single env and render frame by frame, stopping at the first `done`
    or as soon as a frame fails to render. The latter matters here: the policy drives the sim
    to NaN (the dominant failure), and rendering a non-finite state corrupts the viewer's scene
    geoms — so we keep the good lead-up footage and bail before the blow-up frame. This is more
    robust than `play_policy_mujoco`, which keeps rendering past the blow-up and resets.
    MUST run in a process that has not touched MJX/warp (see `record_fall_video`).
    Returns the saved video path, or None if nothing was recorded.
    """
    OmegaConf.set_struct(config, False)
    ep = dict(config.experiment.env_params)
    ep["headless"] = True                                  # offscreen render for recording
    # disable goal visualization: drawing the mimic-site goal geoms overflows the viewer's
    # scene geom buffer in this env and crashes rendering (loco-mujoco viewer.py:399)
    gp = dict(ep.get("goal_params") or {})
    gp["visualize_goal"] = False
    ep["goal_params"] = gp
    # recorder writes to <path>/<tag>/<video_name>.mp4
    venv = factory.make(**ep, custom_dataset_conf=custom_conf,
                        recorder_params=dict(path=str(out_path.parent),
                                             tag=out_path.stem, video_name="clip"))

    network = agent_conf.network
    ts = agent_state.train_state
    if config.experiment.n_seeds > 1:
        ts = jax.tree.map(lambda x: x[0], ts)
    params, run_stats = ts.params, ts.run_stats

    def policy(obs, key):
        (pi, _), _ = network.apply({"params": params, "run_stats": run_stats},
                                   obs, mutable=["run_stats"])
        return pi.mean() if deterministic else pi.sample(seed=key)

    rng = jax.random.key(0)
    obs = venv.reset()
    rendered = 0
    for _ in range(n_steps):
        rng, k = jax.random.split(rng)
        action = jnp.atleast_2d(policy(obs, k))
        obs, _, _, done, _ = venv.step(action)
        try:
            venv.render(record=True)
        except Exception:                                 # non-finite frame -> stop cleanly
            break
        rendered += 1
        if bool(np.asarray(done).ravel()[0]):
            break
    venv.stop()
    return getattr(venv, "_video_file_path", None) if rendered else None


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Diagnostic graphs for a trained PPO agent.")
    ap.add_argument("--path", type=str, default=None, help="Path to PPOJax_saved.pkl")
    ap.add_argument("--n_envs", type=int, default=64)
    ap.add_argument("--n_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample actions instead of using the policy mean")
    ap.add_argument("--video", action="store_true",
                    help="Also record a CPU-MuJoCo clip of a fall to artifacts/figures/agent_debug/")
    ap.add_argument("--video_steps", type=int, default=300,
                    help="Number of control steps to record for --video")
    ap.add_argument("--record_only", action="store_true",
                    help="Internal: record the fall clip only (run in a clean subprocess by --video)")
    ap.add_argument("--video_out", type=str, default=None,
                    help="Internal: output path for --record_only")
    args = ap.parse_args()

    path = Path(args.path) if args.path else \
        find_newest_agent(paths.TRAINED_AGENTS)
    print(f"Loading agent: {path}")
    agent_conf, agent_state = PPOJax.load_agent(str(path))
    config = agent_conf.config

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    traj = Trajectory.load(paths.TRAJ_ADAPTED)
    custom_conf = CustomDatasetConf(traj=traj)

    # record-only mode: this process must NOT build/run the MJX env (it would corrupt the
    # render context), so do the recording and exit before anything MJX is touched.
    if args.record_only:
        out = Path(args.video_out) if args.video_out else (OUTPUT_DIR / "fall.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        vp = _record_fall_inproc(config, custom_conf, factory, agent_conf, agent_state,
                                 out, args.video_steps, deterministic=not args.stochastic)
        print(f"VIDEO_PATH={vp}")
        return

    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = True
    env = factory.make(**config.experiment.env_params, custom_dataset_conf=custom_conf)

    # joint limit bands (read from the MuJoCo model via plot_joint_angles)
    from plot_joint_angles import model_joint_ranges
    ranges = model_joint_ranges()

    act_low = np.asarray(env.info.action_space.low)
    act_high = np.asarray(env.info.action_space.high)
    dt = float(env.dt)

    # active termination thresholds (height band + orientation), read off the env so the
    # "why they die" plot reflects the real terminal condition rather than hardcoded values
    thresholds = read_terminal_thresholds(env)
    lo, hi = thresholds["height_range"]
    rot_thr = thresholds["rot_threshold"]
    rot_str = (f"{np.rad2deg(rot_thr):.1f} deg" if rot_thr is not None
               else "(no orientation threshold)")
    print(f"Terminal handler: {thresholds['handler']}  "
          f"height_range=({lo:.3f}, {hi:.3f}) m  rot_threshold={rot_str}")

    print(f"Rolling out {args.n_envs} envs x {args.n_steps} steps "
          f"({'stochastic' if args.stochastic else 'deterministic'})...")
    data = rollout(env, agent_conf, agent_state, args.n_envs, args.n_steps,
                   deterministic=not args.stochastic, seed=args.seed)

    # alive mask: 1 until (and including) the step that terminates, 0 after
    done = data["done"].astype(bool)
    alive = (np.cumsum(done, axis=0) - done) == 0   # True up to & incl. first done
    alive = alive.astype(float)
    t = np.arange(args.n_steps) * dt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{path.parent.name}_{stamp}"

    plot_joint_tracking(data, alive, t, ranges, OUTPUT_DIR / f"joint_tracking_{tag}.pdf")
    plot_joint_error(data, alive, OUTPUT_DIR / f"joint_error_rms_{tag}.pdf")
    plot_root_tracking(data, alive, t, OUTPUT_DIR / f"root_tracking_{tag}.pdf")
    plot_survival_reward(data, alive, t, dt, OUTPUT_DIR / f"survival_reward_{tag}.pdf")
    plot_reward_breakdown(data, alive, t, OUTPUT_DIR / f"reward_breakdown_{tag}.pdf")
    plot_actions(data, alive, t, act_low, act_high, OUTPUT_DIR / f"actions_{tag}.pdf")
    horizon = int(getattr(env.info, "horizon", args.n_steps))
    traj_len = int(env.th.len_trajectory(0))   # control-step length of the reference clip
    _cause, counts = plot_termination_analysis(data, t, dt, thresholds, horizon, traj_len,
                                               OUTPUT_DIR / f"termination_{tag}.pdf")

    # optional: record a MuJoCo clip so the failure mode can be watched directly
    video_path = None
    if args.video:
        print(f"Recording a {args.video_steps}-step MuJoCo clip of a fall (subprocess)...")
        try:
            video_path = record_fall_video(path, OUTPUT_DIR / f"fall_{tag}.mp4",
                                           args.video_steps, args.stochastic)
        except Exception as e:                            # rendering is best-effort
            print(f"  video recording failed ({type(e).__name__}: {e}).")
            print("  Tip: on a headless box set the GL backend, e.g. "
                  "`MUJOCO_GL=egl python debug_agent.py --video`.")

    # quick text summary
    surv = (~done.any(axis=0)).mean()
    mean_len = (first_done_index(done).mean()) * dt
    jerr = np.rad2deg(np.sqrt(((data["qpos"][:, :, QPOS_JOINT_SLICE]
                                - data["ref_qpos"][:, :, QPOS_JOINT_SLICE]) ** 2).mean()))
    # deaths = everything except "survived" (index 4): instab+height+orient+timeout+wrap
    n_dead = max(int(counts.sum() - counts[4]), 1)
    print("\n================ SUMMARY ================")
    print(f"  survived full episode : {surv*100:.0f}% of {args.n_envs} robots")
    print(f"  mean episode length   : {mean_len:.2f} s  (horizon {args.n_steps*dt:.2f} s)")
    print(f"  mean per-step reward   : {data['reward'].mean():.3f}")
    print(f"  overall joint RMS error: {jerr:.1f} deg")
    print(f"  cause of death (of {n_dead} that ended early):")
    print(f"      traj end (clip ran out): {counts[5]:3d}  ({100*counts[5]/n_dead:.0f}%)  <- benign truncation")
    print(f"      instability (NaN)      : {counts[0]:3d}  ({100*counts[0]/n_dead:.0f}%)")
    print(f"      height (fall/sink)     : {counts[1]:3d}  ({100*counts[1]/n_dead:.0f}%)")
    print(f"      orientation (tip)      : {counts[2]:3d}  ({100*counts[2]/n_dead:.0f}%)")
    print(f"      timeout (horizon)      : {counts[3]:3d}  ({100*counts[3]/n_dead:.0f}%)")
    if video_path:
        print(f"  fall video saved to   : {video_path}")
    print(f"\nGraphs saved to: {OUTPUT_DIR}")
    print("=========================================")


if __name__ == "__main__":
    main()
