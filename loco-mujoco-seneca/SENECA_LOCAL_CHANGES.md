# SENECA local changes to loco-mujoco-seneca

This file tracks local modifications made to this (editable-install) third-party
package for the SenecaBot DeepMimic project, so they can be identified and, if
needed, reverted or ported. Every code change below is also marked in-source with
a `[SENECA LOCAL CHANGE]` comment. Newest first.

---

## 2026-09-29 (later) — Task reward for slopes, gravity-relative falls, PPO critic warm-up

**Files:** `loco_mujoco/core/reward/slope_locomotion.py` (new) and `reward/__init__.py` (registration);
`loco_mujoco/core/terminal_state_handler/upright.py` (new) and its `__init__.py`;
`loco_mujoco/algorithms/ppo_jax.py` (`_train_fn` / `_loss_fn`).

**Why.** Fine-tuning v140 for slopes with its DeepMimic reward keeps it replicating the flat-ground
reference relative to the slope surface: trunk parallel to the ground, the reference joint angles and
1.40 m/s, with a fall defined as the trunk leaving the reference orientation by 30°. Goats walk differently
on steep ground, so the slope fine-tune (seneca_loco `conf_finetune_slope_task.yaml`) replaces the reward
with a task reward and the fall criterion with one that does not use the reference.

**What changed.**
- `SlopeLocomotionReward`: no imitation term. Progress along the fall line
  `exp(-((v_x - target_speed)^2 + v_y^2) / vel_sigma^2)`, an alive bonus, and penalties for yaw rate,
  slope-normal bouncing, trunk roll/pitch rates, lateral tilt w.r.t. gravity (the trunk pitch is free),
  mechanical power, torque², action rate/jerk (as in `MimicReward`) and foot slip while touching the
  ground, plus an air-time term at each touchdown (short swings, i.e. micro-hops, are penalized). Foot
  velocities come from `cvel`/`subtree_com` (stateless, exact); contact uses a height threshold with
  hysteresis for the air time. The total is clipped at 0. Its constructor has no `**kwargs`, so a
  misspelled weight raises. Default weights were calibrated on v140's recorded flat gait: ~1.6 per step,
  of which ~20% are penalties.
- `UprightTerminalStateHandler`: terminal when the trunk's up axis is more than `max_tilt_deg` (50) from
  gravity-up (the episode's gravity, from `SlopeRandomizer`) or the root is below `min_height` (0.15 m).
- PPO `critic_warmup_updates` (experiment config, default 0 = unchanged): during the first N updates the
  actor loss and the entropy bonus are multiplied by 0, so only the critic learns. This lets the loaded
  critic fit a new reward before its advantages move the policy.
- XLA workaround: `SlopeRandomizerState.gravity_dir` stores the unit gravity (`(-sin s, 0, -cos s)`),
  and the observation, the reward and the fall criterion read it through
  `utils/math.gravity_direction()` instead of computing `g / norm(g)`. In jax 0.7.1 XLA compiled that
  3-vector normalization into a Triton block-level fusion. Its deduplicated copy was launched with the
  launch dimensions of a plain loop kernel (48×128 threads for a kernel compiled for 1 warp), so the
  first execution of the 2048-env task fine-tune failed with `CUDA_ERROR_INVALID_VALUE` ("Failed to add
  kernel node to a CUDA graph", or "Failed to launch CUDA kernel: fusion_NNNN" with command buffers
  off). This happened on three launches and was located by dumping the optimized HLO. The 64-env smoke
  tests and the imitation fine-tune did not hit it: fusion decisions depend on the whole graph.

**Verified.** With the v140 policy on 64 MJWarp envs with random slopes, the reward is finite, ~1.5 per
step on flat ground and lower on steeper slopes in both directions (0.74 at 20–30° uphill, 0.67 at
20–30° downhill). Critic warm-up: see the seneca_loco change log.

**Note.** v140 falls within ~0.7 s under CPU MuJoCo even with its own config (it walks under MJWarp), so
the CPU path (`eval.py --use_mujoco`) cannot evaluate these agents. This is not caused by the changes above.

---

## 2026-09-29 — Restore the `loco_mujoco/datasets` subpackage (was dropped by `.gitignore`)

**Files:** `.gitignore` (removed the `loco_mujoco/datasets/` line); `loco_mujoco/datasets/` (5 `.py`
files, 112 KB, no data).

**Why.** The `.gitignore` inherited from upstream ignores `loco_mujoco/datasets/`. Upstream still tracks
those files because they were committed before the rule, but when this package was copied without its
git history into the Definicion-Senecabot repository the rule dropped them. `ImitationFactory` imports
`loco_mujoco.datasets.humanoids.LAFAN1` at load time, so no environment could be built from a fresh
clone (`ModuleNotFoundError: loco_mujoco.datasets`).

**What changed.** Copied `loco_mujoco/datasets/` from upstream robfiras/loco-mujoco at commit
`3921fedb` (2026-03-10, latest at the time; the exact commit this fork started from is unknown) and
removed the ignore rule so git tracks it. Every `loco_mujoco.datasets` symbol the package imports
(`imitation_factory.py`, `smpl/retargeting.py`) resolves. Downloaded datasets never go here (they go to
the paths set in `LOCOMUJOCO_VARIABLES.yaml` or to the Hugging Face cache), so un-ignoring it does not
track data.

---

## 2026-09-29 — Slope walking: `SlopeRandomizer`, `SlopeProjectedGravity`, SenecaBot `gravity_obs`

**Files:** `loco_mujoco/core/domain_randomizer/slope.py` (new) and `__init__.py` (registration);
`loco_mujoco/core/observations/base.py` (`SlopeProjectedGravity` + its `ObservationType` entry);
`loco_mujoco/environments/quadrupeds/senecabot.py` (`gravity_obs` kwarg);
`loco_mujoco/core/visuals/video_recorder.py` (ffmpeg fallback).

**Why.** Fine-tune the v140 agent to walk up and down slopes, with the policy perceiving the slope
(seneca_loco `conf_finetune_slope.yaml`). Slope tests showed v140 is blind to the slope and falls beyond
~20–25° uphill and ~25–30° downhill.

**What changed.**
- `SlopeRandomizer` (domain randomizer): samples one slope per episode, uniform in `slope_range_deg`
  (+ uphill: the robot walks toward +x), and applies it in `update()` as the tilted gravity
  `|g|·(-sin s, 0, -cos s)` over the flat floor. Per-env `opt.gravity` works under MJWarp (a batched
  model field). The world frame is the slope frame, so observations, reward and terminal handler stay
  ground-relative and the flat reference remains valid. Optional linear curriculum on the max |slope|
  (`curriculum_start_deg`, `curriculum_steps` in steps per env; the per-env counter lives in the state
  and is not reset between episodes, so freshly reset envs such as PPO validation sample the start range).
- `SlopeProjectedGravity` (stateful observation): unit gravity in the trunk frame (IMU-like), taken from
  the `SlopeRandomizer` state (falls back to `model.opt.gravity`). The existing `ProjectedGravityVector`
  hard-codes world gravity = -z and cannot see a tilted gravity.
- `SenecaBot(gravity_obs=False)`: when True, appends `SlopeProjectedGravity("proj_gravity", "root")`
  after the default observations (before the goal). Default False: agents trained without it keep their
  358-dim layout and load unchanged.
- `VideoRecorder.stop()`: uses the ffmpeg binary of `imageio-ffmpeg` when there is no system ffmpeg
  (the bare `"ffmpeg"` raised `FileNotFoundError`, which lost the end-of-training W&B video).

**Verified.** 64 MJWarp envs with random slopes: the observation equals the expected value at reset
(error 6e-8) and each env's physics follows its own slope (correlation slope vs forward speed −0.98 with
the v140 policy; speeds match the fixed-slope tests).

**Caller-side.** seneca_loco `simulation/training/finetune.py`, `conf_finetune_slope.yaml`, `train.py`.

---

## 2026-06-10 — Guard episode-mean metrics against 0/0 → NaN (wandb gaps)

**Files:** `loco_mujoco/algorithms/ppo_jax.py` (`_update_step`, the `metric = SummaryMetrics(...)`
block) and `loco_mujoco/utils/metrics.py` (`MetricsHandler.__call__`).

**Why.** Both `mean_episode_return`/`mean_episode_length` were computed as
`jnp.sum(jnp.where(done, returns, 0.0)) / jnp.sum(done)`. When a window has **no** episode
terminations — an update window (`num_steps`, train) or a validation window (`validation.num_steps`)
— `jnp.sum(done) == 0` and the division is `0/0 = NaN`. wandb records NaN as a **gap** ("no data"),
so the curves look like they aren't logging. This is plausible under `learnable_std: true` with no
std floor, where a collapsing policy can produce windows with zero terminations.

**What changed (no PPO-math change; logging/metric values only).** Compute `n_done = jnp.sum(done)`
once, clamp the denominator with `safe_denom = jnp.maximum(n_done, 1.0)` (so the not-taken branch of
`jnp.where`, which JAX still evaluates, is NaN-free), and return the mean only when `n_done > 0`,
else `0.0`. Identical pattern in both call sites so train and validation means agree.

**Caller-side.** None. Best-checkpoint selection in `seneca_loco/.../training.py` already uses
`np.nanargmax`; a zero-done validation now scores `0.0` (low) instead of `NaN`, which that selection
handles the same way.

---

## 2026-06-09 (later 3) — Use env-step as wandb's INTERNAL step; drop the `global_step` custom axis

**File:** `loco_mujoco/algorithms/ppo_jax.py` (`_update_step` → `_live_log`)

**Why.** With the `global_step` custom step-metric (set caller-side via
`define_metric("*", step_metric="global_step")`), **live monitoring of a running run showed an empty
panel** — "Mean Episode Return: There's no data for the selected runs. Try a different X axis
setting. Current X axis: global_step". Verified against the wandb **server** (public API): a *running*
run returns **0** history rows for any key, while *finished* runs with the identical setup return full
history (`16-25-18`: 488 train + 24 validation rows; `15-32-12`: 258) and `global_step` works there as
an axis. So nothing was lost — wandb only materialises a **custom** step-metric axis when the run
**finishes**; in-progress it doesn't render, so you can't watch training live on it. (The default
"Step" = internal-step axis *does* render live.)

**What changed (no PPO-math change).** `_live_log` now calls `wandb.log(data, step=int(step))` with the
**env-step as wandb's internal step**, and no longer emits a `global_step` metric field. The default
"Step" axis is therefore the true env-step **and** renders live. This is the pre-`(later)` scheme
(`step=env_step`) — which previously broke because the caller logged validation **post-hoc** at low
env-steps after the live stream had already advanced the internal step (non-monotonic → dropped). That
constraint is gone: validation is now streamed live in the **same** `wandb.log` call as each update's
train metrics (see `(later 2)`), so the env-step passed as `step=` is monotonically non-decreasing
across every call.

**Caller-side.** `seneca_loco/.../training.py` removes the two `define_metric` lines and switches its
`n_seeds > 1` post-hoc loop to `run.log(data, step=gstep)` (one combined row per update). See that
repo's change log. Supersedes the `(later)` "global_step custom axis" entry below.

---

## 2026-06-09 (later 2) — Stream the FULL validation metrics live (robustness)

**File:** `loco_mujoco/algorithms/ppo_jax.py` (`_update_step` → `_live_log`)

**Why.** Until now the single-seed live callback streamed only the train metrics + `Metric for
Sweep`; the full validation block (`Validation Info/*`, `Validation Measures/*`) was logged by the
caller (`seneca_loco/.../training.py`) **post-hoc, in one burst right before `wandb.finish()`**.
Decoding two real runs showed that burst is fragile: run `2026-06-09/16-25-18` streamed its 488
live train points fine (all acked, last filestream `200 OK` at 17:05:52) but the connection then
stopped acking, so the final batch (the 24 validation rows + the video row, sent 17:07:21–46) and
the `.mp4` media upload **never reached the server** — even though all of it is intact in the local
run dir. The sibling run `15-32-12` (identical code) happened to keep its connection and synced
everything. So end-of-run-only logging loses an entire run's validation + video to any brief sync
hiccup in the last seconds; the live-streamed metrics survive because they are acked incrementally.

**What changed (no PPO-math change).** `_live_log` now also emits, on each validation update
(`is_val`), the complete validation block carrying the **same `global_step`** as that update's train
row: `Metric for Sweep`, `Validation Info/Mean Episode Return`/`Length`, and every
`Validation Measures/<measure>/<quantity>` term (iterating `fields()` over the `ValidationSummary` /
`QuantityContainer`, filtering `size > 0` — keys identical to the old post-hoc loop, verified). The
whole `validation_metrics` dataclass is passed through the existing `io_callback` (it is the zero
container on non-validation updates — same pytree structure — and is only read when `is_val`, so it
passes through safely; verified that `io_callback` reconstructs the `@struct.dataclass` host-side).

**Consequences.** Validation is now acked incrementally during the run (not one final burst), is
interleaved with training in **monotonic** step order (no backward `global_step` jump / right-edge
cram on the default Step axis), and appears **live** instead of only at end-of-run.

**Caller-side.** `seneca_loco/.../training.py` drops its single-seed post-hoc validation loop (now a
no-op `pass`) and wraps the end-of-run video step in `try/except` so a render/upload failure can no
longer skip `wandb.finish()`. The `n_seeds > 1` post-hoc branch is unchanged (the live callback is
disabled when the train fn is vmapped). See that repo's change log.

---

## 2026-06-09 (later) — Live wandb logging no longer hijacks the internal step (use `global_step` axis)

**File:** `loco_mujoco/algorithms/ppo_jax.py` (`_update_step` → `_live_log`)

**Why.** The live single-seed callback logged with `wandb.log(data, step=int(max_timestep))`,
i.e. it set wandb's *internal* step to the env-step count (growing to ~`total_timesteps`).
That works for the live train metrics (logged in increasing order during the run), but it
broke the **post-hoc validation metrics** logged by the caller (`seneca_loco/.../training.py`)
*after* training: by then the internal step was already at ~`total_timesteps`, and wandb
forbids a non-monotonic internal step, so validation could not be placed at its true env-step.
It was therefore put on a separate `validation_step` axis and ended up jammed at the right edge
of the default "Step" view (verified by decoding a run's `.wandb`: the 10 validation records sat
at internal steps `299827200..299827209` while their real env-steps were `29.9M..299M`; the first
even merged with the final training record). Symptom: "metrics stamped on the x-axis with the
wrong timesteps."

**What changed (single line of behavior, no PPO-math change):** `_live_log` now puts the env-step
in a **regular metric field** `data["global_step"] = int(step)` and calls `wandb.log(data)` with
**no** `step=`, letting wandb auto-increment its internal step. The caller declares `global_step`
as a shared custom x-axis (`run.define_metric("*", step_metric="global_step")`), so the live train
metrics and the post-hoc validation metrics overlay on the same true env-step axis. `Metric for
Sweep` (logged live on validation updates) is unaffected — it already carried the right step and now
carries `global_step` too.

**Caller-side.** `seneca_loco/simulation/training/training.py` adds the `define_metric` setup right
after `wandb.init` (before `train_fn` runs, since this callback logs immediately) and logs the
post-hoc validation metrics with `global_step` instead of `validation_step`. See that repo's change log.

---

## 2026-06-09 — Optional fixed model camera for recorded video (`record_camera`)

**File:** `loco_mujoco/core/visuals/viewer.py` (class `MujocoViewer`)

**Why.** The viewer only supported its three procedural camera modes
(`static`/`follow`/`top_static`) built on a free `MjvCamera`; it ignored the XML
`<camera>` elements entirely. SenecaBot defines named tracking cameras in
`senecabot_loco.xml` (`follow`, `follow_west`) and wants the recorded training video
(the wandb "Agent Video", produced by `PPOJax.play_policy` → `mjx_render` →
`parallel_render`) shot from one of them.

**What changed (additive, opt-in — default `record_camera=None` reproduces original
behavior exactly):**

1. New `__init__` kwarg `record_camera=None`. When a name is given it is resolved to a
   camera id via `mj_name2id(..., mjOBJ_CAMERA, ...)` (raises `ValueError` if missing)
   and stored as `self._record_camera_id`; resolved **before** the initial
   `_set_camera()` so the lock applies from frame 0.
2. `_set_camera` early-returns when `self._record_camera_id >= 0`, setting the
   `MjvCamera` to `mjCAMERA_FIXED` / `fixedcamid = record_camera_id`. This overrides
   the static/follow/top_static cycle, so even an interactive keypress keeps the fixed
   camera. With `record_camera=None` the field is `-1` and every code path is unchanged.

**Plumbing.** The kwarg arrives via the env's `**viewer_params` (same path as
`headless`): caller sets `experiment.env_params.record_camera` in `conf.yaml`, which
`ImitationFactory.make(**kwargs)` forwards to the env and on to `MujocoViewer`. In
`parallel_render` the scene camera is driven by env 0's data, so a `mode="track"`
model camera follows env 0's tracked body.

---

## 2026-06-05 — Opt-in exclusion of unobservable root x,y from `MimicReward` qpos tracking

**File:** `loco_mujoco/core/reward/trajectory_based.py` (class `MimicReward`)

**Why.** `MimicReward.qpos_dist` tracked the root free joint's absolute world x,y, but
the SenecaBot policy observes the root via `FreeJointPosNoXY` (no world x,y). Penalizing
a quantity the policy cannot observe is unlearnable and rewards a "stand still" optimum.
The earlier attempt to fix this from config via `joints_for_mimic` (listing only the 12
hinges, excluding `root`) **crashed** at `quat_in_qpos = np.concatenate(quat_in_qpos)`
(`ValueError: need at least one array to concatenate`), because the class unconditionally
assumes the root free joint is in the mimic set (its quaternion is always concatenated and
used in `quaternion_angular_distance`). So the exclusion is done here instead, narrowly.

**What changed (additive, opt-in — default reproduces original behavior exactly):**

1. In `MimicReward.__init__`, right after `self._quat_in_qpos` is built: read
   `self._track_root_xy = kwargs.get("track_root_xy", True)` and precompute a static numpy
   boolean mask `self._qpos_pos_mask`:
   - `track_root_xy=True`  (default) → `~self._quat_in_qpos` (original: x,y,z + joints).
   - `track_root_xy=False` → also drops the root's world x,y (keeps z + orientation + joints).

2. In `MimicReward.__call__`, the qpos position distance now uses the mask:
   `qpos[self._qpos_pos_mask]` instead of `qpos[~self._quat_in_qpos]`.

**Safety.** `_qpos_pos_mask` is a static numpy bool array (host side); indexing the traced
jax array with it under `jit` is the same pattern the file already uses for
`~self._quat_in_qpos` and `~self._free_joint_qvel_mask`, so output shapes stay static.
The quaternion/orientation term (`qpos_quat`, `quaternion_angular_distance`) is untouched.
With `track_root_xy=True` (the default) every other environment using `MimicReward` is
byte-for-byte unchanged — only SenecaBot opts in.

**How to use (conf `reward_params`):**
```yaml
      track_root_xy: false   # default True keeps the original x,y tracking
```
Note: this replaces the broken `joints_for_mimic` approach — do not list the hinges under
`joints_for_mimic` to exclude the root, it will crash.

---

## 2026-06-05 — Explicit root forward-velocity tracking reward in `MimicReward`

**File:** `loco_mujoco/core/reward/trajectory_based.py` (class `MimicReward`)

**Why.** The trained SenecaBot agent converged (by ~13–38M steps) to a "stand
still and balance" policy that never walks forward (base-x stuck ~0.3 m vs.
reference ~1.6 m). `MimicReward` had no term rewarding forward locomotion: the
only root-velocity signal was diluted inside `qvel_dist` (3 of 18 components,
soft exponent). Joint-angle tracking alone is a weak driver of locomotion under
open-loop torque control. This change adds a direct, **observable** incentive to
match the reference root velocity (the policy sees `dq_root`, so it is learnable),
mirroring the math already used by `TargetVelocityTrajReward`.

**What changed (two insertions, additive — no existing behavior altered when the
new weight is left at its default of 0.0):**

1. In `MimicReward.__init__`, after the free-joint qvel mask setup:
   - `self._free_joint_qpos_ind` — root free-joint qpos indices.
   - `self._rootvel_w_sum = kwargs.get("rootvel_w_sum", 0.0)` — weight (default 0 = off).
   - `self._rootvel_w_exp = kwargs.get("rootvel_w_exp", 1.0)` — exponent.

2. In `MimicReward.__call__`, just before the total-reward assembly:
   - Computes local-frame root velocity `(vx, vy, yaw_rate)` for both the current
     `data` and the reference `traj_data_single`, rotated into the root frame.
   - `rootvel_reward = exp(-rootvel_w_exp * mean((vel - vel_ref)^2))`, nan-safe.
   - Adds `self._rootvel_w_sum * rootvel_reward` to `total_reward`.

**How to use (config side, in `seneca_loco/simulation/training/conf.yaml` under
`experiment.env_params.reward_params`):**
```yaml
      rootvel_w_sum: 0.25   # 0.0 disables this term (default)
      rootvel_w_exp: 1.0
```
Both keys are forwarded to the reward constructor via `reward_cls(self, **reward_params)`.

**Backward compatibility.** Default `rootvel_w_sum=0.0` reproduces the original
reward exactly, so other environments/experiments using `MimicReward` are unaffected.

---

## 2026-09-27 — `MimicReward`: contact-pattern reward + once-applied smoothness penalties

**Why.** Agents trained with the rootvel reward (runs 2026-09-26 19-22-10 / 20-48-32) learned
"micro-hops": 2–3 touchdowns per gait cycle on one foot where the reference has one, and hind feet
that barely lift. Measured contact agreement with the reference was 61% / 73% (hind feet 40–75%).
Also, the legacy `action_rate_coeff` is multiplied by its coefficient twice in `__call__`
(`coeff * (coeff * -norm)`), so its effective weight is `coeff**2` (0.0025 → 6e-6, a no-op).

**What changed (additive — all new weights default to 0.0, reproducing the old reward exactly):**

1. `MimicRewardState` gains `last_last_action` (init zeros, shifted each step).
2. New kwargs in `MimicReward.__init__`:
   - `contact_w_sum` (0.0), `contact_sites` (foot site names), `contact_height` (0.037 m),
     `contact_ref_frac` (0.3).
   - `action_rate_w` (0.0): `w * sum((a_t - a_{t-1})^2)`, applied once.
   - `action_jerk_w` (0.0): `w * sum((a_t - 2a_{t-1} + a_{t-2})^2)`, applied once.
3. `MimicReward.init_from_traj` precomputes the reference per-foot contact flags: a foot is in
   stance when its site z is in the lowest `contact_ref_frac` of its (2nd–98th percentile) height
   range in the reference (the kinematic reference's front feet never reach the floor, so an
   absolute threshold can't be used).
4. In `__call__`: `contact_reward = mean(agent_contact == ref_contact)` with
   `agent_contact = site_z <= contact_height`, added as `contact_w_sum * contact_reward`; the two
   smoothness terms are subtracted from the total (outside the legacy -1 penalty clip).
5. `environments/base.py` and `environments/humanoids/base_skeleton.py` `load_trajectory` now also
   call `self._reward_function.init_from_traj(self.th)` (it was never called before; the base
   `Reward.init_from_traj` is a no-op, so other rewards are unaffected).

Verified: `seneca_loco/simulation/analysis/debug_agent.py` component breakdown (extended with
`c_contact` / `c_smooth`) reconstructs the env reward to <1e-6 under both the old and new config,
on MJX/Warp; the numpy (CPU MuJoCo) backend also runs.

---

## 2026-09-27 — `mujoco_mjx.py`: full per-env reset with mjwarp (fixes permanently "stuck" envs)

**Symptom.** In runs 2026-09-26 19-22-10 (unique-yogurt-4) and 20-48-32 (worthy-dragon-5) the logged
`Mean Episode Length` fell in ONE update from ~910 to ~630 (at 25.8M / 36.0M steps) and stayed there
until the end (300M), while return/length kept rising smoothly and the final agents, evaluated
separately, never fall (80% reach the 1000-step horizon, 20% end at the trajectory wrap).

**Cause.** `_mjx_reset_in_step` with `use_mjwarp=True` only zeroed `qpos`/`qvel`. `qacc_warmstart`,
`qacc` and all derived fields carried over from the ended episode, so after a single physics blow-up
the NaN warm-start re-poisoned every new episode: that env ended on every step forever (length-1
episodes). One such env among 2048 (~200 length-1 episodes per update vs ~450 real ones) gives
exactly the ~630 plateau. Reproduced by injecting NaN into one env's qvel: 250/250 following steps
were `done`.

**Fix.** `_mjx_reset_in_step` also restores every public per-env Data field (`_WARP_RESET_FIELDS`)
from `self._first_data`, the same data the initial `mjx_reset` uses. Same test: 1 `done`, then the env
runs normally. Rollouts and the MimicReward breakdown are unchanged (recon error < 1e-6).
