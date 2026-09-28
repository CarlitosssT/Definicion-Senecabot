# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LocoMuJoCo is an imitation learning benchmark for whole-body locomotion control, supporting 12 humanoid and 4 quadruped environments with motion capture datasets (AMASS, LAFAN1). It supports both CPU simulation (MuJoCo) and GPU-accelerated parallel simulation (MJX/MjWarp via JAX).

## Commands

**Install:**
```bash
pip install -e .
# GPU JAX support (optional):
pip install jax["cuda12"]
# SMPL/AMASS support (optional):
loco-mujoco-myomodel-init
```

**Run tests:**
```bash
pytest --ignore=tests/test_task_factories.py
# Single test file:
pytest tests/test_reward.py
```

**CLI utilities (dataset management):**
```bash
loco-mujoco-download           # Download all datasets
loco-mujoco-download-real      # Download real mocap datasets
loco-mujoco-download-perfect   # Download perfect/synthetic datasets
loco-mujoco-set-amass-path     # Configure AMASS path
loco-mujoco-set-lafan1-path    # Configure LAFAN1 path
```

## Architecture

### Simulation backends (`loco_mujoco/core/`)
- `mujoco_base.py` — CPU simulation base class wrapping raw MuJoCo
- `mujoco_mjx.py` — GPU simulation extending base with JAX/vmap/jit support
- `observations/`, `reward/`, `control_functions/`, `domain_randomizer/`, `initial_state_handler/`, `terminal_state_handler/`, `terrain/`, `visuals/`, `wrappers/` — pluggable modular components

### Environments (`loco_mujoco/environments/`)
- `base.py` — `LocoEnv` class extending `MujocoMjx`; adds trajectory following, imitation learning, and dataset integration. This is the main base class all environments inherit from.
- Humanoid and quadruped subclasses live in subdirectories here.

### Task factories (`loco_mujoco/task_factories/`)
- `ImitationFactory` — creates imitation learning tasks with dataset configs
- `RLFactory` — creates plain RL tasks
- Main entry point for users creating environments programmatically

### Algorithms (`loco_mujoco/algorithms/`)
- JAX/Flax implementations of PPO, GAIL, AMP, DeepMimic
- Common utilities: `networks.py`, `dataclasses.py`, `base_algorithm.py`

### Trajectories & datasets (`loco_mujoco/trajectory/`, `loco_mujoco/datasets/`)
- `TrajectoryHandler` manages motion capture data loading and sampling
- Trajectory metrics: DTW, discrete Fréchet distance
- Dataset classes for AMASS, LAFAN1, and native datasets

### Key design patterns
- **Modular plug-in architecture**: reward functions, observations, control types, domain randomization are all interchangeable components passed at environment construction.
- **JAX-first**: Flax dataclasses, `vmap`, `jit` used throughout for performance. Stateful sim wrapped in `SimulationState` dataclasses for functional programming compatibility.
- **Dual backend**: same environment API works on both CPU (MuJoCo) and GPU (MJX); the backend is selected at construction time.

### Public API
`loco_mujoco/__init__.py` exports the top-level API. `loco_mujoco/core/README.md` and `loco_mujoco/environments/README.md` contain usage examples and environment status tables.

## Notes
- CI runs on pushes to `master` and `dev` branches using Python 3.11.
- `tests/test_task_factories.py` is excluded from CI runs.
- Hydra is used for configuration management in training examples (`examples/training_examples/`).

## Change Log

> Newest first. Date, what changed, which files, and *why*. Keep concise and factual.

### 2026-06-09 (later 3) — Env-step as wandb internal step; drop the `global_step` custom axis

- **What.** `ppo_jax.py::_live_log` now logs `wandb.log(data, step=int(env_step))` and no longer emits
  a `global_step` field. The default "Step" axis becomes the true env-step and renders live.
  `[SENECA LOCAL CHANGE]` (source marker + `SENECA_LOCAL_CHANGES.md`).
- **Why.** A custom step-metric axis (`global_step`) only materialises on the wandb server when a run
  **finishes** — so live monitoring of a running run showed "no data on global_step" even though the
  data was logging fine (confirmed via the public API: running run → 0 history rows; finished runs with
  the same setup → full history). Putting the env-step back as wandb's internal step fixes live viewing.
  Safe now because validation streams live in the same call (see `(later 2)`), so `step=env_step` is
  monotonic (the old post-hoc validation that forced the custom axis is gone).
- **Caller-side.** `training.py` drops the two `define_metric` calls and switches the `n_seeds>1`
  post-hoc loop to `run.log(data, step=gstep)`. Supersedes `(later)`. (Files:
  `loco_mujoco/algorithms/ppo_jax.py`.)

### 2026-06-09 (later 2) — Stream the full validation metrics live (robustness fix)

- **What.** `ppo_jax.py::_update_step::_live_log` now also streams the complete validation block
  (`Metric for Sweep`, `Validation Info/*`, all `Validation Measures/<measure>/<quantity>`) on each
  validation update, carrying that update's `global_step`. The whole `validation_metrics` dataclass
  is passed through the existing `io_callback` (zero container on non-validation updates → same
  pytree structure; only read when `is_val`). `[SENECA LOCAL CHANGE]` (source marker +
  `SENECA_LOCAL_CHANGES.md`).
- **Why.** Validation was previously logged caller-side **post-hoc in one burst before
  `wandb.finish()`**. Forensics on run `2026-06-09/16-25-18` showed that burst (24 validation rows +
  the video) and the `.mp4` upload were lost when the wandb sync stopped acking in the last seconds
  of the run, while the incrementally-acked live train curves survived. Streaming validation makes
  every point acked during the run, interleaved in monotonic step order (no right-edge cram), and
  visible live instead of only at end-of-run. No PPO-math change.
- **Caller-side.** `seneca_loco/.../training.py` drops the single-seed post-hoc validation loop and
  hardens the video step (`try/except`) so `wandb.finish()` always runs. `n_seeds > 1` post-hoc
  branch unchanged (no live callback under vmap). (Files: `loco_mujoco/algorithms/ppo_jax.py`.)

### 2026-06-09 (later) — Live wandb callback uses a `global_step` metric instead of `step=`

- **What.** `ppo_jax.py::_update_step::_live_log` no longer passes `step=int(max_timestep)` to
  `wandb.log`; it logs the env-step as a regular field `global_step` and lets wandb auto-increment
  its internal step. `[SENECA LOCAL CHANGE]` (source marker + `SENECA_LOCAL_CHANGES.md` entry).
- **Why.** Setting the internal step to the env-step count advanced it to ~`total_timesteps` by the
  end of the run, so the caller's **post-hoc validation metrics** could not be logged at their true
  env-step (wandb forbids a non-monotonic internal step). They were shunted to a `validation_step`
  axis and jammed at the right edge of the default "Step" view (decoded from a run's `.wandb`: the 10
  validation records sat at internal steps 299827200..299827209 vs real env-steps 29.9M..299M; the
  first merged with the final training row). Logging `global_step` as a shared custom x-axis (declared
  caller-side via `run.define_metric("*", step_metric="global_step")`) makes train + validation overlay
  on the true env-step axis. No PPO-math change; `Metric for Sweep` unaffected. (Files:
  `loco_mujoco/algorithms/ppo_jax.py`.) See `seneca_loco`'s change log for the caller-side edits.

### 2026-06-09 — Optional fixed model camera for recorded video (`record_camera`)

- **What.** `MujocoViewer.__init__` gains an opt-in `record_camera=None` kwarg. When set
  to an XML `<camera>` name, the viewer renders from that camera (`mjCAMERA_FIXED`,
  resolved via `mj_name2id`) instead of the procedural `static`/`follow`/`top_static`
  free-camera modes; `_set_camera` early-returns to keep the lock. `None` =
  byte-for-byte unchanged. (Files: `loco_mujoco/core/visuals/viewer.py`.)
- **Why.** SenecaBot's recorded training video (wandb "Agent Video") should be shot from
  the model's named `follow` camera; the viewer previously ignored XML cameras entirely.
- **Caller-side.** `seneca_loco` sets `experiment.env_params.record_camera: follow` in
  `conf.yaml`; it reaches the viewer via the env's `**viewer_params` (same path as
  `headless`). Marked `[SENECA LOCAL CHANGE]`; see `SENECA_LOCAL_CHANGES.md`.

### 2026-06-08 (later) — Re-add live `Train/Entropy` to the PPO live callback

- **Motivation.** Policy entropy was dropped from the 2026-06-08 live logging (below) as
  "uninformative". The user wants it back as a live exploration / std-collapse early-warning
  signal (the signal that would have flagged the 2026-06-07 collapse).
- **Change (`loco_mujoco/algorithms/ppo_jax.py`, `_update_step` only — no PPO-math change).**
  After the update-epoch scan, take `mean_entropy = loss_info[1][2].mean()` (mean of the loss
  aux's entropy term over epochs×minibatches) and add it to the existing single-seed live
  `io_callback` as `Train/Entropy`. Logged **every update** (~366×/run ≈ one point per 30-60s of
  wall-clock) — no throttle needed: it piggybacks on the callback that already runs per update for
  `Mean Episode Return`/`Length`, so it adds one float and **zero** extra host syncs / training-speed
  cost. Still gated to `n_seeds == 1` (vmapped multi-seed has no live callback).
- **Caveat.** With `learnable_std: false` (the SenecaBot default) the policy std is a fixed
  constant, so the curve is **flat** (sanity check only). It only moves / flags a collapse when
  `learnable_std: true` (a deliberate training-dynamics change made caller-side, not here).

### 2026-06-08 — Real-time wandb logging + expose checkpoint buffer in `PPOJax`

- **Motivation.** Metrics were only logged *after* the entire run: `_train_fn` runs the whole
  training as a single `jax.lax.scan`, stacks per-update metrics, and returns them; the caller
  (`seneca_loco/.../training.py`) logged them in a post-hoc loop. So wandb showed nothing live and
  a crashed run lost all history — which blocked watching the 2026-06-07 policy collapse happen.
- **Changes (`loco_mujoco/algorithms/ppo_jax.py`, `_update_step` only — no PPO-math change).**
  - Added a `jax.experimental.io_callback(..., ordered=True)` (`ordered` keeps callbacks in update
    order so the wandb step stays monotonic) that streams to wandb every update: `Mean Episode
    Return` and `Mean Episode Length`, plus `Metric for Sweep` (= euclidean
    `site_rpos+site_rrotvec+site_rvel`) on validation updates only. The callback lazily imports
    `wandb`, is a no-op if `wandb.run is None`, and is **gated to `config.n_seeds == 1`** (with >1
    seed the train fn is vmapped and the callback args would be batched). The **full validation
    metrics are NOT streamed** — the caller logs them post-hoc at end of training (see below).
  - (Entropy was briefly streamed as `Train/Entropy` but removed: with a state-independent Gaussian
    policy it's just a function of `log_std` and sat ~constant near 6, so uninformative.)
  - `_train_fn` now also returns `"train_state_buffer": runner_state[3]` — the existing buffer that
    already snapshots train states at each validation interval (previously carried then dropped).
    This lets the caller pick a **best checkpoint** (by validation return) instead of the final,
    possibly-collapsed state, with no change to the scan internals.
- **Caller-side use** is in `seneca_loco/simulation/training/training.py`: best-checkpoint
  selection; live train metrics for single-seed; validation metrics logged post-hoc on a custom
  `validation_step` x-axis (so they don't fight the advanced live step). See that repo's change log.
