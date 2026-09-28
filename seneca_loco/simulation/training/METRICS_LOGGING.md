# Metric logging — how every wandb metric is produced

How training/validation metrics for the SenecaBot PPO runs are computed, where in the
code, at what frequency, and what to watch out for when reading them.

> Repos: training entry is `seneca_loco/simulation/training/training.py`; the PPO loop and
> wrappers live in the editable `loco-mujoco-seneca` package
> (`loco_mujoco/algorithms/ppo_jax.py`, `loco_mujoco/core/wrappers/mjx.py`,
> `loco_mujoco/utils/metrics.py`).

---

## TL;DR

- There are **two families**: cheap **training** metrics (every update) and expensive
  **validation** metrics (every validation interval, ~10× per run).
- **Everything is computed *inside* the single `jax.lax.scan`** that is the whole training
  run. "Logged at the end" only means *when wandb receives it*, not when it is computed.
- Single-seed runs (`n_seeds == 1`, our default) **stream** train return/length (+ sweep at
  validation steps) live; validation Info/Measures are **re-emitted post-hoc** on a separate
  `validation_step` x-axis.

---

## Frequency & where-logged table

| Metric (wandb key)                         | Computed in                              | How often                         | Sent to wandb            |
|--------------------------------------------|------------------------------------------|-----------------------------------|--------------------------|
| `Mean Episode Return`                      | `ppo_jax.py::_update_step` (~L340)        | **every update** (~366×)          | **live** (in-scan `io_callback`) |
| `Mean Episode Length`                      | `ppo_jax.py::_update_step` (~L341)        | every update                      | live                     |
| `Train/Entropy`                            | `ppo_jax.py::_update_step` (`loss_info[1][2].mean()`) | every update (~366×)  | live (single-seed only)  |
| `global_step` (the shared env-step x-axis) | `ppo_jax.py::_update_step` (~L343, as `max_timestep`) | every update          | live (as the `global_step` field) |
| `Metric for Sweep`                         | from validation euclidean distances      | every validation step (~10×)      | **live** at validation steps |
| `Validation Info/Mean Episode Return`/`Length` | `metrics.py::MetricsHandler.__call__` (~L139) | every `validation_interval` (~10×) | **post-hoc**, `validation_step` axis |
| `Validation Measures/<measure>/<quantity>` | `metrics.py::MetricsHandler.__call__` (~L214) | every `validation_interval` (~10×) | **post-hoc**, `validation_step` axis |

`num_updates = total_timesteps // num_steps // num_envs` (e.g. `150e6 // 200 // 2048 ≈ 366`).
`validation_interval = num_updates // validation.num` (e.g. `366 // 10 = 36`), so validation
fires `validation.num` (=10) times. Both are derived in `PPOJax.init_agent_conf`
(`ppo_jax.py` ~L80-87).

---

## A. Training metrics — every update

### 1. Per-step counters (the `LogWrapper`)
`LogWrapper` wraps the env and accumulates per-episode return/length on **every env step**.
`loco_mujoco/core/wrappers/mjx.py` ~L137-149:

```python
new_episode_return = state.metrics.episode_returns + reward     # running sum this episode
new_episode_length = state.metrics.episode_lengths + 1
Metrics(
    episode_returns          = new_episode_return * (1 - done),  # reset to 0 at episode end
    episode_lengths          = new_episode_length * (1 - done),
    returned_episode_returns = old * (1 - done) + new_episode_return * done,  # latched at done
    returned_episode_lengths = old * (1 - done) + new_episode_length * done,
    timestep                 = state.metrics.timestep + 1,       # NEVER reset → global counter
    done                     = done)
```

- `returned_episode_*` updates **only when an episode finishes** (`* done`); between finishes
  it holds the last completed episode's value.
- `timestep` is never reset → it is the per-env global step counter.

### 2. Reduction to logged scalars
Once per `_update_step`, over the `num_steps`(200) × `num_envs`(2048) transitions collected in
`traj_batch`. `loco_mujoco/algorithms/ppo_jax.py` ~L340-344:

```python
metric = SummaryMetrics(
    mean_episode_return = jnp.sum(jnp.where(done, returned_episode_returns, 0.0)) / jnp.sum(done),
    mean_episode_length = jnp.sum(jnp.where(done, returned_episode_lengths, 0.0)) / jnp.sum(done),
    max_timestep        = jnp.max(timestep * config.num_envs),
)
```

**Mean Episode Return = average return over only the episodes that *terminated* during this
update's window** (Σ returns at `done` ÷ number of `done`s). `max_timestep` = total env-steps
consumed so far → the wandb x-axis.

### 3. `Train/Entropy`
Mean policy entropy over the update's optimization, taken from the loss aux:
`mean_entropy = loss_info[1][2].mean()` (the `entropy` term of `(total_loss, (value_loss,
loss_actor, entropy))`, averaged over `update_epochs × num_minibatches`). Streamed by the same
live `io_callback` as return/length (single-seed only), so it costs no extra host sync.

For a diagonal Gaussian, `entropy = const + Σ log(std)`, so it's a direct readout of the policy's
exploration noise. **With `learnable_std: false` (conf default) the std is fixed → the curve is a
flat constant** (sanity check only). With `learnable_std: true` it drifts and a sharp monotonic
drop = std collapse (the leading indicator of the kind of policy collapse seen on 2026-06-07).

---

## B. Validation metrics — every `validation_interval` updates

Computed **inside training**, gated by a `lax.cond`. `ppo_jax.py` ~L388-392:

```python
validation_metrics = jax.lax.cond(counter % config.validation_interval == 0,
                                  _evaluation_step,        # run a full eval rollout
                                  mh.get_zero_container)   # otherwise return a zero-filled container
```

### The eval rollout (`_evaluation_step`, ~L346-386)
- reset `validation.num_envs`(100) fresh envs,
- `scan` `validation.num_steps`(100) steps with **stochastic** actions (`pi.sample`, ~L358),
- collect every `env_state`, then call `mh(env_states)`.

### What `MetricsHandler.__call__` computes (`metrics.py` ~L135-216)
1. Same return/length scalars as training, but over the eval rollout (~L139-145) →
   `Validation Info/Mean Episode Return` & `Length`.
2. For each requested **quantity** (`JointPosition→qpos`, `JointVelocity→qvel`,
   `RelSitePosition→site_rpos`, `RelSiteOrientation→site_rrotvec`, `RelSiteVelocity→site_rvel`,
   …) it pulls the **agent's** values and the **reference trajectory's** values at the matching
   frames, then applies three **distance measures** (~L214-216):

```python
euclidean_distance        = mean( euclidean(agent, reference) ),
dynamic_time_warping      = mean( dtw(agent, reference) ),
discrete_frechet_distance = mean( frechet(agent, reference) ),
```

→ logged as `Validation Measures/<measure>/<quantity>`. Example:
`Validation Measures/euclidean_distance/qpos` = mean Euclidean distance between agent and
reference **joint angles** over the rollout. **Lower = better tracking.** Quantities not
requested are `jnp.empty(0)` and skipped.

### Metric for Sweep
`Metric for Sweep = euclidean_distance.site_rpos + site_rrotvec + site_rvel` — the Optuna
objective proxy. Only defined on validation steps. Logged **live** at validation steps for
single-seed runs.

---

## C. Computed vs logged (the key distinction)

**All metrics, including all 10 validations, are computed during the run** inside the
`jax.lax.scan`, stacked into arrays, and returned by `_train_fn`. The only thing that differs
is **when wandb receives them**:

- Train return/length (+ sweep) → pushed **live** from inside the scan (`io_callback`,
  `ppo_jax.py::_update_step`) so curves appear during the run.
- Validation Info/Measures → re-emitted from the already-computed arrays **after** `train_fn`
  returns (`training.py` post-hoc block).

**The x-axis.** Every metric carries the env-step in a regular field **`global_step`**, declared
once as a shared custom step metric right after `wandb.init`
(`run.define_metric("*", step_metric="global_step")`); nothing passes `step=` to `wandb.log`, so
wandb's internal step just auto-increments. This is why the live train curves and the post-hoc
validation metrics overlay on the **same** true env-step axis. (Earlier code instead set wandb's
internal step to the env-step via `step=max_timestep`; that hijacked the monotonic internal step so
the post-hoc validation metrics couldn't be placed at their real env-step and jammed at the right
edge of the default "Step" view — fixed 2026-06-09, see the change logs in both repos.)

For `n_seeds > 1` the live callback is disabled (the train fn is vmapped) and everything is
logged post-hoc, also carrying `global_step`.

---

## D. Gotchas (important for interpretation)

1. **NaN source.** Both train and validation returns use `Σ(returned)/Σ(done)`. If **no
   episode terminates** in the window → `0/0 = NaN`; and if physics goes NaN, reward → return
   → metric goes NaN. This is why some validation evals come back NaN (MjWarp
   non-determinism) — and why best-checkpoint selection must use `nanargmax`, not `argmax`.
2. **Return is RAW, not normalized.** `LogWrapper` sits *inside* `NormalizeVecReward`
   (`ppo_jax.py::_wrap_env` ~L527-530), so it accumulates the env's raw reward. PPO trains on
   the normalized reward, but **Mean Episode Return is the true reward sum** — directly
   interpretable as the imitation reward.
3. **Only terminated episodes count.** An update where most envs are still mid-episode
   contributes only the few that finished → noisy early-training return/length.
4. **Validation is stochastic and capped at 100 steps.** `pi.sample` (not deterministic) +
   `num_steps=100`, so validation length can never exceed 100 and is **not** directly
   comparable to the training horizon of 1000. The end-of-run **video** is the only
   deterministic rollout (`play_policy(deterministic=True)`).
5. **Validation ≠ training conditions.** Fresh resets (RSI random gait phase), 100 envs, 100
   steps — a different distribution than the 2048-env/200-step training rollout. Part of why
   the train/val gap can look large.

---

## E. Where the saved checkpoints come from

- During training, the full TrainState is snapshotted into `train_state_buffer` at every
  validation interval (`ppo_jax.py` ~L179, ~L404-407) — `validation.num` (=10) snapshots.
- At the end, `training.py` picks the snapshot with the highest **validation return**
  (`np.nanargmax`) → saves it as `PPOJax_saved.pkl` (the "best"); the final state is saved as
  `PPOJax_final.pkl`. Only these two of the 10 snapshots reach disk.
