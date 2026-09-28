"""
thesis_eval_figures.py

Generate the *evaluation* figures for the thesis (§3.4 Agent Evaluation / Results)
from the trained agents, as publication-quality vector PDFs.

It computes, for each agent, the same metric families the project already defines
(no new definitions are invented here — the scalars reuse `agent_diagnostics`, the
15 tracking distances reuse the framework's `MetricsHandler`):

  1A. Validation distance measures — 5 quantities x 3 measures = 15 tracking
      distances, exactly as the framework's `MetricsHandler` computes them during
      training validation (JointPosition / JointVelocity / RelSitePosition /
      RelSiteOrientation / RelSiteVelocity  x  Euclidean / DTW / Discrete Frechet).
  1B. Per-joint RMS angle error [deg]  (joint_rms_mean / _worst / _max + per-joint).
  1C. Base-height tracking RMS [m].
  2.  Locomotion outcomes — speed_ratio, surv, eplen_frac, bad_death_frac, act_rate.
  3.  Composite score (Optuna v3 weighting) per agent, plus the Optuna study's
      optimization history and FANOVA parameter importances.
  +   Cost of Transport (CoT) and a foot-contact / gait-phase diagram, computed from
      a dynamics-augmented rollout (actuator power and per-foot floor contact).

Two rollouts per agent:
  * deterministic 64 x 300 rollout (debug_agent.rollout, the *verified* one) -> 1B,
    1C, 2, 3, CoT, phase. Matches debug_agent / agent_diagnostics byte-for-byte.
  * stochastic 100 x 100 validation rollout -> the 15 distances, mirroring how
    training measured them (pi.sample, LogWrapper+VecEnv, MetricsHandler).

Usage (from seneca_loco/, conda env `workspace`):
    MUJOCO_GL=egl python -m simulation.analysis.thesis_eval_figures            # compute + plot
    python -m simulation.analysis.thesis_eval_figures --plot-only              # re-plot from cache
    python -m simulation.analysis.thesis_eval_figures --agents 4_best_try      # subset

Outputs (vector PDF + a PNG preview) land in artifacts/figures/thesis_eval/.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pickle
from pathlib import Path

import numpy as np

from simulation.config import paths

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
GRAVITY = 9.81
QVEL_JOINT_SLICE = slice(6, 18)          # 12 hinge-joint generalized velocities (after 6-DoF root)

OUT_DIR = paths.FIGURES / "thesis_eval"
RECORDS_PKL = OUT_DIR / "records.pkl"

# The 5 curated agents, in training-progression order (label -> pkl glob under curated/).
CURATED = {
    "1_good_try": "1_good_try/PPOJax_saved.pkl",
    "2_best_try": "2_best_try/21-16-20/PPOJax_saved.pkl",
    "3_best_try": "3_best_try/10-16-51/PPOJax_saved.pkl",
    "4_best_try": "4_best_try/17-19-49/PPOJax_saved.pkl",
    "5_best_try": "5_best_try/07-58-27/PPOJax_saved.pkl",   # headline agent for single-agent figures
}
BEST_LABEL = "5_best_try"

# How agents are labelled in the figures (the headline agent is just "trained agent").
DISPLAY_NAMES = {"5_best_try": "trained agent"}


def _disp(label):
    return DISPLAY_NAMES.get(label, label)

# The 15 tracking distances, in display order.
QUANTITIES = ["qpos", "qvel", "site_rpos", "site_rrotvec", "site_rvel"]
QUANTITY_LABELS = {
    "qpos": "JointPosition", "qvel": "JointVelocity", "site_rpos": "RelSitePosition",
    "site_rrotvec": "RelSiteOrientation", "site_rvel": "RelSiteVelocity",
}
MEASURES = ["euclidean_distance", "dynamic_time_warping", "discrete_frechet_distance"]
MEASURE_LABELS = {
    "euclidean_distance": "Euclidean", "dynamic_time_warping": "DTW",
    "discrete_frechet_distance": "Discrete Fréchet",
}


# ==============================================================================
# COMPUTE
# ==============================================================================
def build_env(path):
    """Build the imitation env for a saved agent (mirrors agent_diagnostics._load_and_eval)."""
    from omegaconf import OmegaConf
    from loco_mujoco import TaskFactory
    from loco_mujoco.algorithms import PPOJax
    from loco_mujoco.trajectory.dataclasses import Trajectory
    from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf

    agent_conf, agent_state = PPOJax.load_agent(str(path))
    config = agent_conf.config
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    traj = Trajectory.load(paths.TRAJ_ADAPTED)
    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = True
    env = factory.make(**config.experiment.env_params,
                       custom_dataset_conf=CustomDatasetConf(traj=traj))
    return env, agent_conf, agent_state, config


def validation_distances(env, agent_conf, agent_state, config, *, n_envs=100, n_steps=100, seed=0):
    """The 15 framework tracking distances, computed the way training validation does:
    stochastic actions, LogWrapper+VecEnv, then MetricsHandler over the collected states.
    Returns {measure: {quantity: float}} plus mean_episode_return / _length."""
    import jax
    from loco_mujoco.algorithms import PPOJax
    from loco_mujoco.utils import MetricsHandler

    mh = MetricsHandler(config, env)
    wenv = PPOJax._wrap_env(env, config.experiment)

    ts = agent_state.train_state
    if config.experiment.n_seeds > 1:
        ts = jax.tree.map(lambda x: x[0], ts)
    network = agent_conf.network
    params, run_stats = ts.params, ts.run_stats

    rng = jax.random.key(seed)
    obsv, env_state = wenv.reset(jax.random.split(rng, n_envs))

    def step(carry, _):
        env_state, last_obs, rng = carry
        rng, k = jax.random.split(rng)
        (pi, _), _ = network.apply({"params": params, "run_stats": run_stats},
                                   last_obs, mutable=["run_stats"])
        action = pi.sample(seed=k)
        obsv, _, _, _, _, env_state = wenv.step(env_state, action)
        return (env_state, obsv, rng), env_state

    (_, _, _), env_states = jax.lax.scan(step, (env_state, obsv, rng), None, length=n_steps)
    vm = mh(env_states)

    out = {m: {} for m in MEASURES}
    for m in MEASURES:
        container = getattr(vm, m)
        for q in QUANTITIES:
            out[m][q] = float(getattr(container, q))
    out["mean_episode_return"] = float(vm.mean_episode_return)
    out["mean_episode_length"] = float(vm.mean_episode_length)
    return out


def cost_of_transport(data, mass, dt):
    """Dimensionless cost of transport from a dynamics-augmented rollout.

    CoT = E / (m g d), with mechanical energy E = integral of total positive-and-negative
    actuator power |tau . qdot| over each env's alive window, and d the forward distance the
    base actually travels in that window. Averaged over the envs that move (excludes envs that
    die immediately or barely translate). Also returns the mean instantaneous power trace [W].
    """
    from simulation.analysis.debug_agent import first_done_index
    done = data["done"].astype(bool)
    T, N = done.shape
    alive = ((np.cumsum(done, axis=0) - done) == 0).astype(float)          # (T, N)

    qfrc = data["qfrc_actuator"][:, :, QVEL_JOINT_SLICE]                    # (T, N, 12)
    qd = data["qvel"][:, :, QVEL_JOINT_SLICE]                              # (T, N, 12)
    power = np.abs(qfrc * qd).sum(axis=-1)                                  # (T, N) [W]
    energy = (power * alive * dt).sum(axis=0)                              # (N,) [J]

    end = np.clip(first_done_index(done) - 1, 0, T - 1)
    ar = np.arange(N)
    dist = np.linalg.norm(data["qpos"][end, ar, 0:2] - data["qpos"][0, :, 0:2], axis=-1)  # (N,)

    ok = (end >= 10) & (dist > 1e-2)
    cot = energy[ok] / (mass * GRAVITY * np.maximum(dist[ok], 1e-6))
    cot_mean = float(np.nanmean(cot)) if ok.any() else float("nan")

    power_trace = np.where(alive.sum(1) < 0.5, np.nan,
                           (power * alive).sum(1) / np.maximum(alive.sum(1), 1e-6))
    return cot_mean, power_trace


def gait_phase(data, env_index=None):
    """Per-foot floor-contact timeline (fr, fl, br, bl) for ONE representative env (the
    longest-surviving by default). Returns (foot_contact (T, 4), t [s], chosen env index)."""
    from simulation.analysis.debug_agent import first_done_index
    done = data["done"].astype(bool)
    T = done.shape[0]
    if env_index is None:
        env_index = int(np.argmax(first_done_index(done)))                  # longest survivor
    fc = data["foot_contact"][:, env_index, :]                             # (T, 4)
    alive_T = int(first_done_index(done)[env_index])
    return fc[:alive_T], np.arange(alive_T), env_index


def evaluate_one(path, *, n_envs=64, n_steps=300, val_envs=100, val_steps=100, seed=0):
    """Full evaluation record for one agent: validation 15-distances + scalar diagnostics +
    CoT + a representative gait timeline."""
    import mujoco
    from simulation.analysis.debug_agent import rollout, read_terminal_thresholds
    from simulation.analysis import agent_diagnostics as ad

    env, agent_conf, agent_state, config = build_env(path)
    model = env.get_model()
    mass = float(np.sum(model.body_mass))
    ctrl_dt = float(env.dt)          # control timestep (sim dt * n_substeps); one rollout step

    # --- deterministic diagnostics rollout (with dynamics for CoT + contact) ---
    data = rollout(env, agent_conf, agent_state, n_envs, n_steps,
                   deterministic=True, seed=seed, collect_dynamics=True)
    thresholds = read_terminal_thresholds(env)
    horizon = int(getattr(env.info, "horizon", n_steps))
    traj_len = int(env.th.len_trajectory(0))
    scal = ad._scalars_for_one(data, thresholds, horizon, traj_len, worst_k=3)
    composite = ad.composite_from_scalars(scal)
    cot, power_trace = cost_of_transport(data, mass, ctrl_dt)

    # geometric stance detection: foot in contact when its sphere's lowest point ~ floor (z=0)
    from simulation.analysis.debug_agent import FOOT_GEOM_NAMES
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOM_NAMES]
    foot_r = np.array([float(model.geom_size[fid][0]) for fid in foot_ids])     # (4,)
    data["foot_contact"] = (np.asarray(data["foot_z"]) <= foot_r[None, None, :] + 0.012).astype(float)
    fc, fc_t, fc_env = gait_phase(data)

    # --- stochastic validation rollout -> 15 framework distances ---
    val = validation_distances(env, agent_conf, agent_state, config,
                               n_envs=val_envs, n_steps=val_steps, seed=seed)

    return dict(
        scalars={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in scal.items()},
        per_joint_rms=np.asarray(scal["per_joint_rms"]).tolist(),
        composite=float(composite),
        cot=float(cot),
        power_trace=power_trace.tolist(),
        gait_contact=fc.tolist(), gait_t=fc_t.tolist(), gait_env=int(fc_env),
        ctrl_dt=ctrl_dt, mass=mass,
        validation=val,
    )


def compute_all(labels, **kw):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = {}
    if RECORDS_PKL.exists():
        with open(RECORDS_PKL, "rb") as f:
            records = pickle.load(f)
    for label in labels:
        path = paths.TRAINED_AGENTS / "curated" / CURATED[label]
        print(f"\n=== evaluating {label}  ({path.name}) ===")
        try:
            records[label] = evaluate_one(path, **kw)
        except Exception as e:
            # Older curated agents were trained against an earlier fork obs spec; the current
            # fork builds a wider observation, so their normalization layer can't be replayed.
            # Skip and report rather than abort the whole sweep.
            print(f"  [SKIP] {label}: {type(e).__name__}: {e}")
            records.pop(label, None)
            continue
        s = records[label]["scalars"]
        print(f"  speed_ratio={s['speed_ratio']:.3f}  surv={s['surv']:.3f}  "
              f"jmean={s['joint_rms_mean']:.2f}  composite={records[label]['composite']:.3f}  "
              f"CoT={records[label]['cot']:.3f}")
        with open(RECORDS_PKL, "wb") as f:                  # checkpoint after each agent
            pickle.dump(records, f)
    return records


# ==============================================================================
# PLOT  (publication-quality vector PDF)
# ==============================================================================
def _setup_style():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 120, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })
    return plt


def _agent_colors(labels):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    return {lab: cmap(i / max(len(labels) - 1, 1)) for i, lab in enumerate(labels)}


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    print(f"  wrote {name}.pdf / .png")


def fig_validation_distances(records, labels):
    """1A: 3 panels (one per measure), grouped bars: x=quantity, bars=agents."""
    plt = _setup_style()
    colors = _agent_colors(labels)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(QUANTITIES))
    w = 0.8 / len(labels)
    for ax, m in zip(axes, MEASURES):
        for j, lab in enumerate(labels):
            vals = [records[lab]["validation"][m][q] for q in QUANTITIES]
            ax.bar(x + (j - (len(labels) - 1) / 2) * w, vals, w,
                   label=_disp(lab), color=colors[lab])
        ax.set_title(MEASURE_LABELS[m])
        ax.set_xticks(x)
        ax.set_xticklabels([QUANTITY_LABELS[q] for q in QUANTITIES], rotation=35, ha="right")
        ax.set_ylabel("distance (lower = better)")
    axes[-1].legend(title="agent", frameon=False, ncol=1, loc="upper right")
    fig.suptitle("Validation tracking distances — 5 quantities × 3 measures (framework MetricsHandler)",
                 y=1.02)
    fig.tight_layout()
    _save(fig, "1A_validation_distances")
    plt.close(fig)


def fig_per_joint_rms(records, labels):
    """1B: heatmap (agents × 12 joints) + summary bars (mean/worst/max)."""
    from simulation.analysis.debug_agent import JOINT_NAMES
    plt = _setup_style()
    colors = _agent_colors(labels)
    M = np.array([records[lab]["per_joint_rms"] for lab in labels])          # (A, 12)

    fig, (axh, axb) = plt.subplots(1, 2, figsize=(13, 4.4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    im = axh.imshow(M, aspect="auto", cmap="magma")
    axh.set_xticks(np.arange(12)); axh.set_xticklabels(JOINT_NAMES, rotation=60, ha="right")
    axh.set_yticks(np.arange(len(labels))); axh.set_yticklabels([_disp(l) for l in labels])
    axh.set_title("Per-joint RMS angle error [deg]"); axh.grid(False)
    cb = fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04); cb.set_label("RMS [deg]")

    x = np.arange(len(labels)); w = 0.27
    for k, key, lbl in [(-1, "joint_rms_mean", "mean"), (0, "joint_rms_worst", "worst-3"),
                        (1, "joint_rms_max", "max")]:
        axb.bar(x + k * w, [records[lab]["scalars"][key] for lab in labels], w, label=lbl)
    axb.set_xticks(x); axb.set_xticklabels([_disp(l) for l in labels], rotation=30, ha="right")
    axb.set_ylabel("RMS [deg]"); axb.set_title("Joint-error summary"); axb.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "1B_per_joint_rms")
    plt.close(fig)


def fig_locomotion_outcomes(records, labels):
    """1C + 2: small multiples for the outcome scalars."""
    plt = _setup_style()
    colors = _agent_colors(labels)
    specs = [
        ("speed_ratio", "Speed ratio (→1 = on pace)", 1.0),
        ("surv", "Survival fraction (↑)", 1.0),
        ("eplen_frac", "Episode length fraction (↑)", 1.0),
        ("bad_death_frac", "Bad-death fraction (↓)", None),
        ("act_rate", "Action rate / jerk proxy (↓)", None),
        ("height_rms", "Base-height RMS [m] (↓)", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    x = np.arange(len(labels))
    bw = 0.5 if len(labels) == 1 else 0.8
    for ax, (key, title, ref) in zip(axes.flat, specs):
        vals = [records[lab]["scalars"][key] for lab in labels]
        ax.bar(x, vals, width=bw, color=[colors[l] for l in labels])
        if ref is not None:
            ax.axhline(ref, color="k", ls="--", lw=0.9, alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels([_disp(l) for l in labels], rotation=30, ha="right")
        ax.set_xlim(-0.7, len(labels) - 0.3); ax.set_title(title)
    fig.suptitle("Locomotion-outcome metrics" + ("" if len(labels) == 1 else " across curated agents"),
                 y=1.0)
    fig.tight_layout()
    _save(fig, "2_locomotion_outcomes")
    plt.close(fig)


def fig_composite(records, labels):
    """3: composite score per agent (lower = better), with the baseline ≈6.03 line."""
    from simulation.analysis.agent_diagnostics import composite_from_scalars  # noqa
    plt = _setup_style()
    colors = _agent_colors(labels)
    vals = [records[lab]["composite"] for lab in labels]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, vals, width=(0.5 if len(labels) == 1 else 0.8), color=[colors[l] for l in labels])
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.axhline(6.03, color="crimson", ls="--", lw=1.0, label="baseline 13-24-37 (≈6.03)")
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([_disp(l) for l in labels], rotation=30, ha="right")
    ax.set_ylabel("composite score (lower = better)")
    ax.set_title("Optuna-v3 composite score per agent")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "3_composite_score")
    plt.close(fig)


def fig_cost_of_transport(records, labels):
    plt = _setup_style()
    colors = _agent_colors(labels)
    fig, (axb, axp) = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(labels))
    axb.bar(x, [records[lab]["cot"] for lab in labels],
            width=(0.5 if len(labels) == 1 else 0.8), color=[colors[l] for l in labels])
    axb.set_xlim(-0.7, len(labels) - 0.3)
    for xi, lab in zip(x, labels):
        axb.text(xi, records[lab]["cot"], f"{records[lab]['cot']:.2f}",
                 ha="center", va="bottom", fontsize=8)
    axb.set_xticks(x); axb.set_xticklabels([_disp(l) for l in labels], rotation=30, ha="right")
    axb.set_ylabel("CoT  = E / (m g d)"); axb.set_title("Cost of Transport (↓ better)")

    for lab in labels:
        p = np.asarray(records[lab]["power_trace"])
        t = np.arange(len(p)) * records[lab]["ctrl_dt"]
        axp.plot(t, p, color=colors[lab], lw=1.2, label=_disp(lab))
    axp.set_xlabel("time [s]"); axp.set_ylabel("mechanical power [W]")
    axp.set_title("Mean actuator power over the rollout"); axp.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "4_cost_of_transport")
    plt.close(fig)


def fig_cost_of_transport_single(records, label=BEST_LABEL):
    """Standalone CoT figure for a single agent: the per-step mechanical-power trace over the
    deterministic rollout, with the scalar CoT annotated. (A one-bar chart conveys nothing, so
    for a single agent we show the power signal the CoT integral is computed from instead.)"""
    plt = _setup_style()
    colors = _agent_colors([label])
    rec = records[label]
    p = np.asarray(rec["power_trace"])
    t = np.arange(len(p)) * rec["ctrl_dt"]
    cot = rec["cot"]
    mean_p = float(np.mean(p))

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(t, p, color=colors[label], lw=1.3)
    ax.axhline(mean_p, color="0.4", ls="--", lw=1.0, label=f"mean = {mean_p:.1f} W")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mechanical power  $\\sum_j |\\tau_j\\,\\dot q_j|$  [W]")
    ax.set_title(f"Mechanical power over the rollout — {_disp(label)}")
    ax.set_xlim(t[0], t[-1])
    # annotate the resulting dimensionless CoT
    ax.text(0.98, 0.94,
            f"CoT $= E\\,/\\,(m g d) = {cot:.2f}$\n"
            f"$m = {rec['mass']:.2f}$ kg",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "4_cost_of_transport_single")
    plt.close(fig)


def fig_gait_phase(records, label=BEST_LABEL):
    """Foot-contact / gait-phase diagram for one agent (the headline one)."""
    plt = _setup_style()
    rec = records[label]
    fc = np.asarray(rec["gait_contact"])                                   # (T, 4)
    t = np.asarray(rec["gait_t"]) * rec["ctrl_dt"]
    feet = ["FR", "FL", "BR", "BL"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for i, foot in enumerate(feet):
        stance = fc[:, i] > 0.5
        ax.fill_between(t, i + 0.05, i + 0.95, where=stance, step="mid",
                        color="tab:blue", alpha=0.85)
    ax.set_yticks(np.arange(4) + 0.5); ax.set_yticklabels(feet)
    ax.set_ylim(0, 4); ax.set_xlabel("time [s]")
    ax.set_title(f"Foot-contact (stance) diagram — {_disp(label)} (env {rec['gait_env']})")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "5_gait_phase_diagram")
    plt.close(fig)


def fig_optuna(study_db="optuna_phase2_v3.db", study_name="ppo_phase2_rewards_v3"):
    """3 (search): Optuna optimization history + FANOVA parameter importances + slice plots."""
    try:
        import optuna
        from optuna.visualization.matplotlib import (plot_optimization_history,
                                                      plot_param_importances, plot_slice)
    except Exception as e:
        print(f"  [optuna] skipped ({e})")
        return
    plt = _setup_style()
    storage = f"sqlite:///{paths.OPTUNA_DIR / study_db}"
    study = optuna.load_study(study_name=study_name, storage=storage)

    ax1 = plot_optimization_history(study)
    fig1 = ax1.figure; fig1.set_size_inches(7, 4.2)
    ax1.set_title("Optuna Phase-2 optimization history")
    fig1.tight_layout(); _save(fig1, "3b_optuna_history"); plt.close(fig1)

    try:
        import matplotlib.text as mtext
        ax2 = plot_param_importances(study)
        fig2 = ax2.figure; fig2.set_size_inches(7, 4.2)
        # optuna writes the importance-plot title as a standalone Text artist (not ax.title),
        # so set_title would render a SECOND title overlapping it — relabel it in place.
        for _t in fig2.findobj(mtext.Text):
            if _t.get_text() == "Hyperparameter Importances":
                _t.set_text("Reward-weight importance (fANOVA)"); break
        else:
            ax2.set_title("Reward-weight importance (fANOVA)")
        fig2.tight_layout(); _save(fig2, "3c_optuna_importance"); plt.close(fig2)
    except Exception as e:
        print(f"  [optuna importance] skipped ({e})")

    # slice plots: objective vs. each searched reward weight (one panel per parameter)
    try:
        axes3 = np.atleast_1d(plot_slice(study))
        fig3 = axes3.flat[0].figure
        fig3.set_size_inches(13, 6.5)
        for ax in axes3.flat:
            ax.title.set_fontsize(8)
            ax.tick_params(labelsize=6.5)
            ax.xaxis.label.set_size(7.5); ax.yaxis.label.set_size(7.5)
        fig3.suptitle("Optuna Phase-2 slice plots — composite vs. each reward weight", y=1.0)
        fig3.tight_layout(rect=(0, 0, 1, 0.97))
        _save(fig3, "3d_optuna_slice"); plt.close(fig3)
    except Exception as e:
        print(f"  [optuna slice] skipped ({e})")


# ==============================================================================
# WANDB training-time curves (entropy + validation distances over env-step)
# ==============================================================================
WANDB_ENTITY = "andrademarique-universidad-de-los-andes"
WANDB_PROJECT = "deepmimic_senecabot"
HEADLINE_RUN = "faithful-sunset-317"     # = 5_best_try (07-58-27, full 300M-step run)
WANDB_CSV = OUT_DIR / "wandb_history.csv"

_WANDB_KEYS = (["Train/Entropy", "Mean Episode Return", "Mean Episode Length", "Metric for Sweep"]
               + [f"Validation Measures/{m}/{q}" for m in MEASURES for q in QUANTITIES])


def pull_wandb_history(run_name=HEADLINE_RUN, force=False):
    """Download the headline run's logged history (entropy, returns, the 15 validation
    distances) vs env-step, and cache it to CSV so plotting is offline-reproducible."""
    import pandas as pd
    if WANDB_CSV.exists() and not force:
        return pd.read_csv(WANDB_CSV)
    import wandb
    api = wandb.Api(timeout=60)
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"display_name": run_name})
    run = list(runs)[0]
    print(f"  pulling history for {run.name} (id={run.id}, state={run.state})")
    rows = list(run.scan_history(keys=["_step"] + _WANDB_KEYS, page_size=10000))
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(WANDB_CSV, index=False)
    print(f"  cached {len(df)} rows -> {WANDB_CSV.name}")
    return df


def fig_training_curves(run_name=HEADLINE_RUN):
    """Training-time monitoring signals for the headline run: policy entropy, episode return,
    and the 15 validation tracking distances over env-step (Training / Diagnostics figure)."""
    import pandas as pd
    plt = _setup_style()
    df = pull_wandb_history(run_name).sort_values("_step")
    step = df["_step"].to_numpy() / 1e6     # millions of env-steps

    # --- entropy + return ---
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, lab in [(a0, "Train/Entropy", "policy entropy"),
                         (a1, "Mean Episode Return", "mean episode return")]:
        s = df[key]
        m = s.notna().to_numpy()
        ax.plot(step[m], s.to_numpy()[m], lw=1.3, color="tab:blue")
        ax.set_xlabel("env-steps [millions]"); ax.set_ylabel(lab); ax.set_title(lab)
    fig.suptitle(f"Training-time monitoring — {run_name}", y=1.01)
    fig.tight_layout(); _save(fig, "6_training_entropy_return"); plt.close(fig)

    # --- the 15 validation distances over training (3 panels, 5 lines each) ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    cmap = plt.get_cmap("tab10")
    for ax, m in zip(axes, MEASURES):
        for i, q in enumerate(QUANTITIES):
            col = f"Validation Measures/{m}/{q}"
            if col not in df:
                continue
            s = df[col]; mask = s.notna().to_numpy()
            ax.plot(step[mask], s.to_numpy()[mask], "-o", ms=3, lw=1.2,
                    color=cmap(i), label=QUANTITY_LABELS[q])
        ax.set_title(MEASURE_LABELS[m]); ax.set_xlabel("env-steps [millions]")
        ax.set_ylabel("validation distance")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"Validation tracking distances over training — {run_name}", y=1.02)
    fig.tight_layout(); _save(fig, "6_validation_over_training"); plt.close(fig)


def plot_all(records, labels):
    fig_validation_distances(records, labels)
    fig_per_joint_rms(records, labels)
    fig_locomotion_outcomes(records, labels)
    fig_composite(records, labels)
    fig_cost_of_transport(records, labels)
    fig_cost_of_transport_single(records, BEST_LABEL if BEST_LABEL in records else labels[-1])
    fig_gait_phase(records, BEST_LABEL if BEST_LABEL in records else labels[-1])
    fig_optuna()


# ==============================================================================
# CLI
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", nargs="*", default=list(CURATED),
                    help="subset of curated labels (default: all 5)")
    ap.add_argument("--plot-only", action="store_true", help="re-plot from cached records")
    ap.add_argument("--wandb", action="store_true",
                    help="also pull + plot the headline run's training-time curves (needs network)")
    ap.add_argument("--wandb-only", action="store_true", help="only the wandb training curves")
    ap.add_argument("--n_envs", type=int, default=64)
    ap.add_argument("--n_steps", type=int, default=300)
    args = ap.parse_args()

    if args.wandb_only:
        fig_training_curves()
        print(f"\nWandb figures in {OUT_DIR}")
        return

    labels = [a for a in CURATED if a in args.agents]

    if args.plot_only:
        with open(RECORDS_PKL, "rb") as f:
            records = pickle.load(f)
    else:
        records = compute_all(labels, n_envs=args.n_envs, n_steps=args.n_steps)

    plot_labels = [l for l in labels if l in records]
    plot_all(records, plot_labels)
    if args.wandb:
        fig_training_curves()
    print(f"\nAll figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
