# CLAUDE.md

Final-state reference for Claude Code (and humans) working in this repository.

> **Status: project COMPLETE (June 2026).** This file documents the finished system. The
> detailed development change log (2026-06-05 → 2026-06-12, with all diagnoses and retractions)
> was condensed on finalization — see this file's **git history** for the full log. If you do
> make further changes, append a dated entry to the **Change Log** section at the bottom.

---

## Project Context

**Goal.** Train a quadruped robot ("SenecaBot") to imitate a reference walking gait derived from
ovine (sheep) motion-capture data, using DeepMimic-style imitation learning. Undergraduate
thesis project (the LaTeX thesis lives in the sibling `../thesis-doc/` repo; for thesis writing,
read `../thesis-doc/THESIS_CONTEXT.md`, not this code).

**Outcome.** The goal was achieved. Best agent:
`artifacts/trained_agents/curated/4_best_try/17-19-49/PPOJax_saved.pkl` — **speed_ratio 0.985**
(walks at reference pace), survival 0.97, 0 bad deaths, joint RMS 7.40° mean, composite 5.014
(< 6.03 baseline). Trained 100M steps with the anti-shuffle reward weights + gamma 0.995.

**Stack.**
- `loco_mujoco` (imitation-learning benchmark) — installed **editable from the local fork**
  `/home/ndam/Desktop/workspace/loco-mujoco-seneca`, NOT upstream.
- JAX + Flax, PPO (`PPOJax`), GPU-accelerated MJX / MjWarp simulation (11 GB RTX 5070).
- Hydra + OmegaConf for config, Weights & Biases for logging, Optuna for HPO.
- **Conda env: `workspace`** (has `loco_mujoco`, `jax`, `mujoco`). Python ≥ 3.11.

**Cross-repo coupling (critical).** Live wandb logging, best-checkpoint saving, and several
reward features (`rootvel_*`, `track_root_xy`, `qvel_w_exp`, `record_camera`) depend on **local
edits to the fork**, marked `[SENECA LOCAL CHANGE]` in-source and indexed in
`../loco-mujoco-seneca/SENECA_LOCAL_CHANGES.md`. A single feature is often split across
`train.py` here and the fork's `loco_mujoco/algorithms/ppo_jax.py` — check both before
changing training behavior, and log fork changes in *that* repo too.

---

## Architecture: C3D → trajectory → PPO

```
.c3d MOCAP (200 Hz)                                 data/C3D_Final/
  → Stage 1–3  python -m simulation.mocap.run      → data/processed_data/trajectory.npz
  → Stage 4–5  simulation/mocap/trajectory/        → data/processed_data/trajectory_adapted.npz
  → simulation/training/train.py (PPOJax)          → artifacts/trained_agents/<date>/<time>/PPOJax_saved.pkl
  → eval.py / analysis/debug_agent.py / analysis/agent_diagnostics.py
```

**Path discipline:** ALL filesystem paths come from `simulation/config/paths.py` (repo root
overridable via `$SENECA_HOME`; kept in sync with `conf.yaml`'s `hydra.run.dir`). ALL generated
output goes to the gitignored `artifacts/` (`trained_agents/`, `figures/`, `optuna/`, `wandb/`,
`recordings/`). Never recompute `Path(__file__).parent...` chains or write outside `artifacts/`.

**Stages 1–3 — `simulation/mocap/` package** (run from the repo root):
`python -m simulation.mocap.run [--augment] [--file-index N | --c3d PATH]` does C3D→markers
(`io.py`), markers→joint angles (`kinematics.py`), angles→qpos/qvel via MuJoCo FK (`build.py`),
optional augmentation (`augment.py`); prints a time-consistency report.
**Single source of truth** for rates, marker→joint map and layout:
`simulation/config/pipeline.py` (`POINT_RATE = 200` Hz capture, `CONTROL_RATE = 100` Hz env).

**Stages 4–5 — `simulation/mocap/trajectory/trajectory_generation.py`** (Hydra, config from
`simulation/training/`): loads `trajectory.npz` (or `+use_augmented=true`), extracts the best
steady-state gait cycle, **symmetrizes left/right legs** and **tiles it into a ~60 s looping
clip** with accumulated root translation (`simulation/mocap/cyclic.py`; disable with
`+symmetrize_gait=false` / `+make_cyclic=false`), runs standalone-MuJoCo FK for the site/body
quantities LocoMuJoCo needs, and writes `trajectory_adapted.npz` — the training reference.
Mimic site names are auto-read from the XML (sites ending `_mimic`).

**Env.** The `SenecaBot` env class lives **in the fork**
(`loco-mujoco-seneca/loco_mujoco/environments/quadrupeds/senecabot.py`), loading this repo's
`simulation/assets/xml/senecabot_loco.xml` (so that XML dir must not move). qpos =
`[root_pos(3), root_quat(4), 12 hinges]`; joint order `fr_hip, fr_knee, fr_ankle, fl_*, br_*,
bl_*`. Control is **open-loop torque** (`DefaultControl`, ±80 Nm hips/knees, ±40 Nm ankles);
the `position_control` block in `conf.yaml` is dead config (defined but unused).

**Training.** `simulation/training/train.py` (Hydra, reads `conf.yaml` in the same dir —
experiment / env / reward / PPO config) builds the env via the fork's `ImitationFactory` +
`MimicReward` and trains with `PPOJax`. Run dirs land in `artifacts/trained_agents/` (absolute
`hydra.run.dir` — cwd-independent). Saves **`PPOJax_saved.pkl` = best-validation checkpoint**
(what eval/debug default to) and `PPOJax_final.pkl` = final state. Train/validation metrics
stream live to wandb from inside the JAX scan via a fork-side `io_callback` — see
`simulation/training/METRICS_LOGGING.md`.

**Evaluation chain.**
- `simulation/training/eval.py` — replay a saved agent (`--use_mujoco` for CPU sim, `--camera follow`).
- `simulation/analysis/debug_agent.py` — 7 diagnostic figures to `artifacts/figures/agent_debug/`
  (per-joint tracking, RMS, root tracking, survival/reward, reward-term breakdown,
  actions/saturation, termination causes; `--video` records the lead-up to the first fall).
- `simulation/analysis/agent_diagnostics.py` — reduces the same rollout to scalars
  (`speed_ratio`, `joint_rms_*`, `height_rms`, `surv`, `bad_death_frac`, composite). Also the
  Optuna Phase-2 objective. `speed_ratio` ≈ 1.0 = walks at reference pace; composite < 6.03
  beats the 13-24-37 baseline.
- `simulation/mocap/trajectory/trajectory_playback.py` — play back the reference clip (no agent).
- `simulation/analysis/plot_joint_angles.py` — trajectory joint angles vs model joint limits
  → `artifacts/figures/joint_angles_*.png`.

**HPO.** `notebooks/optuna_optimization.ipynb` — Phase 1 tuned PPO knobs (winner folded into
`conf.yaml`); Phase 2 tunes reward weights against the `agent_diagnostics` composite (study
`ppo_phase2_rewards_v3`). SQLite DBs live in `artifacts/optuna/`.

**MuJoCo models (`simulation/assets/xml/`).**
- `senecabot_loco.xml` — the **training** model (free-joint root + 12 hinges, joints named
  `*_joint`, includes `scene.xml`/`mimic_sites.xml`, `follow`/`follow_west` cameras).
- `senecabot_move.xml` — templated model used by movement generation (`scripts/main.py`
  renders a scripted walk via `simulation/movement/walking.py`); joints named `fr_hip`…,
  all 12 properly limited. Used by `plot_joint_angles.py` for limit bands. Templated XMLs are
  loaded with `simulation/core/data_loader.py::load_model` + the geometric `parameters` dict in
  `simulation/config/parameters.py`.
- `senecabot.xml` — older templated model (all-positive ranges; back legs do not match the gait).

**Thesis figures (`final_document/`).** A standalone, self-contained pipeline (own README,
not path-coupled to `simulation/`) that turns raw CSVs in `final_document/data/` into
traceable vectorial PDFs in `final_document/images/`. To add a figure: write a plotter in
`scripts/plotters.py`, register it in `scripts/registry.py`, drop its CSV in `data/`. Global
look lives in `scripts/style.py`. Every output is named `<category>__<descriptor>__<variant>.pdf`
(double underscores — single `_` is a LaTeX math op) and logged to `images/_manifest.tsv` with
its source CSVs, plotter, kwargs hash, and timestamp. Allowed categories: `io_utils.CATEGORIES`.
This is the figure source for the sibling `../thesis-doc/` LaTeX repo.

**Hardware (`hardware/`).** Physical-prototype sensing, independent of the sim: an Arduino
sketch (`potenciometer.ino`) and a `sensing_viewer.ipynb` notebook for reading joint
potentiometers. Not part of the training pipeline.

---

## Commands

All from the **repo root** after `conda activate workspace`:

```bash
python main.py                                                 # interactive menu over all tasks below
python -m simulation.mocap.run [--augment]                     # stages 1–3
python -m simulation.mocap.trajectory.trajectory_generation    # stages 4–5 (+use_augmented=true)
python simulation/training/train.py                            # train (Hydra; conf.yaml)
python simulation/training/eval.py --path artifacts/trained_agents/<d>/<t>/PPOJax_saved.pkl [--use_mujoco] [--camera follow]
python simulation/analysis/debug_agent.py [--path …/PPOJax_saved.pkl] [--n_envs 64] [--n_steps 300] [--stochastic] [--video]
python simulation/analysis/agent_diagnostics.py --path …/PPOJax_saved.pkl   # scalars + composite
python -m simulation.mocap.trajectory.trajectory_playback      # play the reference, no agent
python scripts/main.py                                         # scripted walk (no RL)
jupyter notebook notebooks/optuna_optimization.ipynb           # HPO

# Thesis figures (run from final_document/; standalone pipeline)
cd final_document && python scripts/generate_all.py            # all figures → images/*.pdf
python scripts/generate_all.py "training__*" [--fmt png] [--latex] [--clean]   # filter/format
```

**No test suite here** — tests live in the fork (`cd ../loco-mujoco-seneca && pytest`).

---

## Gotchas

- **Train-time `Mean Episode Length` is not a survival measure** (training-loop artifact reads
  ~637 when the true E[L] ≈ 900) — judge agents with `debug_agent`/`agent_diagnostics` rollouts.
- The true MOCAP capture rate is **200 Hz**, read from the `.npz` — never hardcode 2000 (that
  was the analog/force-plate rate).
- `artifacts/trained_agents/curated/` holds hand-curated runs (`curated/4_best_try/17-19-49` =
  the final best agent); the date dirs are raw Hydra run outputs.
- MjWarp rollouts are **nondeterministic** run-to-run (occasional NaN envs); use `np.nanargmax`
  -style guards when reducing over evals.
- `trajectory_generation.py` shares `conf.yaml`'s Hydra config, so its run logs also land in
  `artifacts/trained_agents/<date>/<time>/` (log-only dirs, no `.pkl`).
- **Notebook edits get clobbered** if a notebook is open in a live Jupyter session when files
  are edited on disk — close/reload it first.
- Control is open-loop torque; the `position_control` PD gains in `conf.yaml` are dead config.

---

## Project history (condensed)

Key milestones, newest first. Full diagnostic detail is in this file's git history.

- **2026-06-12 — Restructure executed** (branch `restructure`): all generated output moved to
  gitignored `artifacts/`, every path centralized in `simulation/config/paths.py`
  ($SENECA_HOME override), the `hydra.run.dir` path-doubling bug fixed (absolute run dir),
  source reorganized (`core/`, `analysis/`, `mocap/trajectory/`, `train.py`, `scripts/`),
  stale `simulation/senecabot.py` deleted, `setup.py` → `pyproject.toml`. Verified: regenerated
  reference bit-identical, GPU smoke train + full evaluation chain pass. Added the root
  `main.py` interactive launcher.
- **2026-06-11 — Goal reached.** Anti-shuffle reward weights (stronger `rootvel`/`qvel` forward
  terms, revived `qvel_w_exp=0.25` gradient, 13-24-37 pose/penalty values) + **gamma 0.995**
  (0.99's ~100-step credit horizon made far-off deaths invisible) produced run `17-19-49`:
  speed_ratio 0.985 vs 0.22 (in-place shuffler) and 0.51 (half-pace). The earlier shuffling was
  traced to Optuna-v2 weights tuned under a speed-blind metric. The ~637 train-time episode
  length was proven a training-loop metric artifact, not deaths (true E[L] ≈ 906–932).
- **2026-06-09/10 — Diagnostics-driven metric + reference fixes.** Left/right gait
  **symmetrization** in the reference (peak 10.4° L/R bias → 0) fixed the front-left trip.
  `agent_diagnostics.py` composite created (v3 adds `height_rms`/`speed_ratio`) and made the
  Optuna Phase-2 objective. wandb logging made live + robust (env-step axis, streamed
  validation, best-checkpoint protection); anti-std-collapse config (`ent_coef 1e-2`,
  `init_std 0.3`, `anneal_lr`).
- **2026-06-06/08 — Root causes found.** "Agent dies" was a **reference-clip-end NaN wrap**,
  not instability — fixed by the looping reference (`cyclic.py`: best gait cycle tiled ~60 s
  with accumulated root translation; survival 0%→97%, joint RMS 29-32°→7.4°). The "reward is
  gamed" finding was retracted (a debug-script reconstruction bug; the joint-position term is
  in fact the largest reward component). The 200 Hz capture-rate labeling bug (was 2000/100)
  fixed half-speed playback. MOCAP pipeline refactored from notebooks into the
  `simulation/mocap/` package. Best-checkpoint saving + live wandb logging added (fork-coupled).
- **2026-06-05 — Project log started**; `debug_agent.py` diagnostic suite created.

---

## Change Log

> Condensed on finalization (2026-06-12). The full 2026-06-05 → 2026-06-12 development log is
> preserved in this file's git history. New post-finalization changes go below, newest first.

### 2026-09-29 (later) — slope fine-tune with a task reward (`conf_finetune_slope_task.yaml`)

- **Why:** the imitation fine-tune (`conf_finetune_slope.yaml`, below) keeps rewarding the flat-ground
  reference relative to the slope surface (posture, joint angles, 1.40 m/s) and ends the episode when the
  trunk stops being parallel to the slope. Goats change gait and posture on steep ground.
- **What:** `train.py --config-name conf_finetune_slope_task` starts from v140 with `SlopeLocomotionReward`
  (progress along the fall line, alive bonus, stability, energy, foot slip, air time, and MimicReward's
  smoothness terms; no imitation term), `UprightTerminalStateHandler` (fall = trunk >50° from gravity-up or
  root <0.15 m) and `critic_warmup_updates: 10`. The same `SlopeRandomizer` ±30° and `gravity_obs` are
  used. v140's reference goal stays in the observation, because the network depends on it as a
  rhythm/phase input, and the reference still gives the initial states; nothing rewards following it.
  `finetune.resolve_config` now drops the init agent's `<kind>_params` when the fine-tune changes
  `<kind>_type` (reward, terminal state, ...), instead of merging MimicReward keys into the new reward.
- **Result:** run `artifacts/trained_agents/2026-09-29/18-49-47` (W&B `hvruinbk`), 100M steps in ~35 min.
  Reward per step rose from 0.69 to 1.55. Final checkpoint, same fall criterion as v140 (trunk >50° from
  vertical): no falls from −35° to +30°, 22% at +35°, where v140 falls from +25° and at −35°. Speed stays
  at 1.33–1.46 m/s on every slope, where v140 goes from 2.15 downhill to 0.56 m/s uphill. Uphill cost of
  transport at 20° is 1.6 against v140's 4.2. The cycle period is unchanged (0.63 s, the reference clock).
  Analysis and figures are in the repo-level `pruebas_pendiente/` (`visualize_slopes.ipynb`).
- **Caveat, checkpoint selection:** `PPOJax_saved.pkl` (best validation return) is effectively random here.
  Validation episodes last 100 steps (1 s), so its return only counts episodes that fell within that
  second (0.0 in 9 of 24 evaluations). Use `PPOJax_final.pkl` or the slope tests to choose.
- `simulation/analysis/debug_agent.py::rollout` skips the MimicReward term breakdown (`c_*` keys) when the
  env's reward is not MimicReward. It used to crash on agents trained with another reward.

### 2026-09-29 — slope fine-tune of v140 (`conf_finetune_slope.yaml`, `finetune.py`)

- **Why:** slope tests (repo-level `pruebas_pendiente/`) showed v140 is blind to the slope and falls
  beyond ~20–25° uphill (grip, then the gait stalls) and ~25–30° downhill (pitches nose-down); the
  actuators never saturate.
- **What:** `python simulation/training/train.py --config-name conf_finetune_slope` fine-tunes from a
  saved agent. `finetune.py` takes the experiment config saved inside the agent's .pkl (not conf.yaml)
  merged with `finetune.experiment`; maps the weights onto the new observation layout by observation
  name, with zero input weights for the added entries, so the start policy and value are exactly v140's
  (asserted); measures the normalization of the added entries with the v140 policy; fresh optimizer.
  The config adds `gravity_obs` (trunk-frame gravity, 3 obs → 361) and `SlopeRandomizer` (±30°, uniform
  per episode; the curriculum is available but off), lr 5e-5, 100M steps, and trains explicitly on the
  1.40 m/s reference (`trajectory_adapted.preRootSpeed.bak.npz`) whatever the active
  `trajectory_adapted.npz` is. New fork pieces in `loco-mujoco-seneca/SENECA_LOCAL_CHANGES.md`.
- `train.py` without a `finetune` block is unchanged.

### 2026-09-27 — active reference switched back to 1.40 m/s (A/B run)

- `data/processed_data/trajectory_adapted.npz` is again the **1.40 m/s** (mocap-speed) reference, for a
  100M run with the new reward terms but the original speed (run 2026-09-27/18-22-14, W&B 88fqa0qs,
  override `rootvel_w_exp=4.0`). The 0.218 m/s no-slip reference is kept as
  `trajectory_adapted.noslip_0218.npz` (agent 2026-09-27/07-12-14 was trained on it); copy it back
  over `trajectory_adapted.npz` (or regenerate with option 2) to use it again.

### 2026-09-27 — fixed the mjwarp in-step reset (episode-length "drop" in training)

- **Why:** unique-yogurt-4 / worthy-dragon-5 showed an abrupt, permanent drop of Mean Episode Length
  (~910 → ~630) and Return at 25.8M / 36.0M. Not a policy regression: the mjwarp in-step reset only
  zeroed qpos/qvel, so one env whose physics blew up kept a NaN `qacc_warmstart` and ended on every
  step forever (length-1 episodes dragging the mean). Reproduced by NaN injection.
- **What:** `loco-mujoco-seneca/loco_mujoco/core/mujoco_mjx.py` restores all per-env Data fields from
  the initial data on reset (see its SENECA_LOCAL_CHANGES.md). Runs before this date have the artifact
  in their W&B return/length curves; their saved agents are not affected.

### 2026-09-27 — reference root speed rescaled to the no-slip speed

- **Why:** the reference root travel was the sheep's T13 marker (1.40 m/s), but with the robot's
  shorter legs each foot sweeps only ~13 cm per 0.635 s cycle, so every stance foot slid at
  1.0–1.3 m/s. The rootvel reward then pushed agents to micro-hop to keep up.
- **What:** `simulation/mocap/cyclic.py` adds `no_slip_root_speed` (FK with the body still;
  speed minimizing stance-foot slip, stance = lowest 30% of foot height, same as `contact_ref_frac`)
  and `scale_root_speed` (scales root XY travel, XY qvel and per-cycle `dxy`; height, orientation,
  joints untouched). `make_looping_reference(root_speed=...)` and `trajectory_generation.py`
  (`+root_speed=no_slip|mocap|<m/s>`, default `no_slip`). Regenerated `trajectory_adapted.npz`:
  root 1.402 → 0.218 m/s (×0.155). Previous file kept as
  `data/processed_data/trajectory_adapted.preRootSpeed.bak.npz`; `+root_speed=mocap` reproduces
  it bit-exactly. Stance-foot world slip dropped from 102–134 to 58–67 cm/s; the remainder is in
  the joint angles themselves (feet move within "stance"), not fixable by root speed.
- **conf.yaml:** `rootvel_w_exp` 4.0 → 100.0 so standing still is still penalized at the new speed
  (0.21 of the term instead of 0.94).
- Agents trained before this date track the old 1.40 m/s reference; compare them with the old file.

### 2026-09-27 — anti-micro-hop reward terms (contact pattern + smoothness)

- **Why:** the 100M/300M rootvel agents (2026-09-26 19-22-10 / 20-48-32) micro-hop (2–3 touchdowns
  per cycle on one foot, hind feet barely lift; 61–73% contact agreement with the reference). The
  legacy `action_rate_coeff` is applied squared inside `MimicReward` (effective 6e-6, a no-op).
- **What:** `MimicReward` (loco-mujoco-seneca, see its `SENECA_LOCAL_CHANGES.md`) gains
  `contact_w_sum` (reward for matching the reference's per-foot stance/swing phase), `action_rate_w`
  and `action_jerk_w` (1st/2nd action-difference penalties, applied once). `conf.yaml` sets all three
  to 0.5 (calibrated on the 300M agent: contact ≈ +0.37/step, smoothness ≈ -0.25/step vs total ≈ 2.1).
  `simulation/analysis/debug_agent.py` breakdown adds `c_contact` / `c_smooth`; its rollout now resets
  the tracked last actions on episode reset, like the env.
- **Caveat:** the reference's root speed (1.42 m/s) exceeds what its leg sweep supports (~0.2 m/s
  no-slip), so the contact term pushes toward feet that stay down while sliding rather than hopping.

### 2026-09-26 — fixed scripted walk (`scripts/main.py`, launcher option 9)

- **Why:** `simulation/movement/walking.py` still read the legacy `*_shoulder_angle` CSV columns,
  but `ovino_angles.csv` names the middle segment `*_knee_angle` (see the rename note in
  `simulation/config/pipeline.py`), so the scripted walk crashed with `KeyError: 'fl_shoulder_angle'`.
- **What:** `ANGLE_COLS` and the per-step actuator reads now use `*_knee_angle`; dropped the now
  redundant `.replace("shoulder", "knee")`. Verified: renders `artifacts/figures/walking_simulation.mp4`.

### 2026-06-23 — comparable reference vs. policy 10 s clips

- **What:** new `simulation/analysis/render_clips.py` renders two 10 s MuJoCo mp4s with an identical
  base-following side camera (mediapy/ffmpeg, EGL): `reference_clip.mp4` (trajectory_adapted from
  step 0) and `policy_clip.mp4` (trained policy, **RSI off** so it resets to the SAME step-0 pose).
  Both → `artifacts/recordings/`, 500 frames @ 50 fps. 5_best_try survived the full 999/1000 steps.
- **Why:** RSI off makes the ideal motion and the learned motion start from the same canonical pose
  so they can be compared frame-for-frame. Verified frame 0 of both is the identical pose.
- Loads the model XML for rendering (no MJX needed there); only the policy rollout builds the env.

### 2026-06-20 (later 16) — "imitation targets" schematic (Idea A)

- **What:** new `simulation/analysis/gait_targets_figure.py` renders SenecaBot in a representative
  mid-stride pose (30 % of a reference gait cycle) and overlays, via an analytic world->pixel
  projection (free-camera, fovy 45), the three kinds of imitation target so the reader sees *what
  the agent tracks*: the 13 mimic SITES colour-coded by group (foot/shank/thigh/base), the 12 hinge
  JOINTS (magenta diamonds, hip/knee/ankle labelled on the front-right leg), and the base site (root
  pose). A legend + reward-term mapping (qpos / root / rpos / rquat / rvel) ties each to subsec 2.4.6.
  Body rendered lightly transparent; markers drawn in matplotlib on top (no occlusion). Output
  `schematic__imitation_targets__v1.{pdf,png}`.
- **NB:** the projection initially mirrored because the free-camera forward sign was flipped; correct
  is `forward = [cos(el)cos(az), cos(el)sin(az), sin(el)]`, `cam_pos = lookat - dist*forward`.
- Copied to `final_document/images/` and `../thesis-doc/assets/` (same name). Loads the model XML
  directly (no MJX/GPU needed); fast CPU/EGL render.

### 2026-06-20 (later 15) — gait-symmetry: `--agent` flag (reference vs. agent)

- **What:** added `--agent <pkl>` to `gait_symmetry.py`. Rolls out the policy (RSI off, single
  representative surviving env, 8×400 deterministic) via `build_env`+`rollout`, folds its qpos into
  a mean cycle, and runs the same symmetry metrics; with the default reference present it prints
  reference-vs-agent and writes `gait_symmetry_agent.{pdf,png}` (agent overlay) +
  `gait_symmetry_compare.{pdf,png}` (per-joint GSI@50 grouped bars).
- **Result (5_best_try):** reference 99.0 % symmetric vs **agent 69.3 %** (GSI@50 0.010 → 0.307).
  Asymmetry concentrates in the **hind limbs** (hind GSI 0.32 vs fore 0.29); the standout is
  **hind_knee**, whose contralateral correlation is ≈ −0.09 and best-fit lag tau* ≈ 1 % (not 50 %)
  — the two hind knees lost antiphase coordination. i.e. the policy did *not* fully preserve the
  reference's bilateral symmetry. (A genuinely discriminating thesis result, unlike the reference,
  which is symmetric by construction.)

### 2026-06-20 (later 14) — bilateral gait-symmetry metric

- **What:** new `simulation/analysis/gait_symmetry.py` — bilateral (left-right) symmetry of a
  processed trajectory. Folds the clip into one mean gait cycle (FR-hip-trough boundaries, circular
  phase grid), then per contralateral joint pair (fr<->fl, br<->bl × hip/knee/ankle) computes the
  phase-shifted symmetry index (best-fit lag tau* via circular RMS minimisation + fixed-50% residual,
  ROM-normalised, + Pearson corr of right vs. half-shifted left) and the Robinson SI on ROM/peak/mean.
  Prints a table; writes `artifacts/figures/thesis_eval/gait_symmetry.{pdf,png}` (per-pair overlay,
  Times New Roman).
- **Finding:** `trajectory_adapted.npz` is symmetric *by construction* — aggregate GSI@50 = 0.010
  (99 % symmetric), tau* = 50 % exactly, corr 1.00, Robinson SI ≈ 0 % on all 6 pairs (left legs are
  the right legs shifted half a stride; the retargeting/augmentation enforces this). So on the
  reference the metric is a sanity check; the discriminating use is on the **agent rollout** (does
  the learned policy preserve symmetry?) — `symmetry_table`/`fold_mean_cycle` are written to accept
  any (N,nq) qpos so an agent rollout can be passed later (use a single representative env / RSI off,
  to avoid the cross-env phase-averaging amplitude loss documented in (later 13)).

### 2026-06-20 (later 13) — joint_tracking_pct_gait regenerated with RSI OFF (amplitude fix)

- **Why:** the reference (and policy) curves in `joint_tracking_pct_gait.pdf` had ~3× too-small
  amplitude vs `joint_angles_trajectory_adapted.pdf` (fr_hip ref swing 6.8° vs the true 20.2°).
  Cause: `plot_joint_tracking` averages joint angles **across envs** (`_alive_mean`) *before*
  folding into a gait cycle; under random-start (RSI) every env is at a different gait phase, so
  averaging the periodic signal across phases cancels the oscillation (same RSI cross-env artifact
  as the root_tracking sawtooth, here as amplitude loss). Per-env the policy/ref amplitudes were
  fine (~20°) — only the figure understated them.
- **What:** regenerated this one figure with RSI disabled so all envs reset to the same trajectory
  phase (step 0), making the cross-env mean preserve full amplitude. Method (one-off, not wired
  into the pipeline): build env, `env.th.random_start = False`, deterministic rollout (8 envs ×
  400 steps), call `plot_joint_tracking` → Debug_data `joint_tracking_07-58-27_20260620_174236.pdf`
  (same filename), copied to `../thesis-doc/assets/joint_tracking_pct_gait.pdf`. Verified ref
  fr_hip swing now 20.2°, matching `joint_angles_trajectory_adapted.pdf`.
- **NB / scope:** RSI was turned off ONLY for this figure. The shared `debug_agent.py` `main()`
  pipeline still uses RSI — correct for the other 6 figures (survival/termination/root stats want
  the randomised-phase population). A more principled in-pipeline fix would be per-env phase-folding
  in `_gait_phase_average` (fold each env on its own troughs, then pool) — not done here per the
  user's choice to just disable RSI for this plot.

### 2026-06-20 (later 12) — qualitative gait visuals (filmstrip + footfall diagram)

- **Why:** the thesis "Qualitative Gait Assessment" had no *rendered* gait and no contact/duty-
  factor diagram (assets only had generic textbook gait graphics).
- **What:** new `simulation/analysis/gait_visuals.py` (offscreen EGL, Times New Roman) →
  `artifacts/figures/thesis_eval/`:
  - `gait_filmstrip.{pdf,png}` — 2×6 grid of MuJoCo-rendered poses across ONE gait cycle (FR-hip-
    trough boundaries), trained agent (top) vs. reference (bottom) at matching phases; camera
    tracks the base so each pose is centred.
  - `gait_footfall_diagram.{pdf,png}` — stance bars + duty factor D per leg (FR/FL/BR/BL),
    phase-averaged over all cycles. **Agent = true simulated ground contact** (foot geom z ≤
    radius+12 mm). **Reference = kinematic foot-height proxy** (lowest 30 % of each foot's own
    z-range): the retargeted reference is a *floating* kinematic target — its front feet hover
    ~6 cm and never physically contact, so an absolute floor threshold reads ~0 % stance. The two
    definitions are labelled distinctly in the legend.
  - Result reads sensibly: agent duty FR/FL/BR/BL ≈ 26/37/80/59 %, reference ≈ 28/28/39/48 %;
    diagonal pairs (FR↔BL, FL↔BR) align between agent and reference, but the agent walks with
    higher duty factors (more grounded) than the reference.
- Both PDFs copied to `../thesis-doc/assets/` (`gait_filmstrip.pdf`, `gait_footfall_diagram.pdf`).
- **NB:** the reference footfall is a *foot-height proxy*, not physical contact — state this in any
  caption; it doubles as evidence that the retargeted clip isn't contact-resolved.

### 2026-06-20 (later 11) — enlarged reward_breakdown panel titles (13→15)

- **What:** bumped the two panel titles in `plot_reward_breakdown` to 15 pt. Regenerated the
  headline agent's `reward_breakdown_07-58-27_20260620_174236.pdf` and copied over
  `../thesis-doc/assets/`.

### 2026-06-20 (later 10) — bigger fonts on reward_breakdown figure

- **What:** bumped type sizes in `plot_reward_breakdown` (panel titles→13, axis labels→12, tick
  labels→11, legend 7→9, per-bar value labels 8→9.5) for thesis legibility. Regenerated the
  headline agent's `reward_breakdown_07-58-27_20260620_174236.pdf` (same filename) and copied over
  `../thesis-doc/assets/`.

### 2026-06-20 (later 9) — bigger fonts on root_tracking figure

- **What:** bumped the type sizes in `plot_root_tracking` (suptitle 14→16, panel titles 9/def→12/13,
  tick labels→11, "time [s]" axis labels→12) for thesis legibility. Regenerated the headline
  agent's `root_tracking_07-58-27_20260620_174236.pdf` (same filename) and copied over
  `../thesis-doc/assets/`.

### 2026-06-20 (later 8) — fixed the base-x/y sawtooth artifact in root_tracking

- **Why:** the base-x panel showed a sawtooth in the reference and a spurious gap below the
  policy (ref ~2.8 m vs policy ~4.2 m at 3 s), making it look like the policy overshot forward
  distance. Root cause: the panel averaged **absolute** world-x across envs. Under random-start
  (RSI) envs begin anywhere in 2..85 m and the reference clip *loops* (world-x snaps ~85 m→0.65 m
  at the boundary); when a wrapped, high-offset env left the alive set the cross-env mean lurched
  ~1.3 m — the sawtooth. (An earlier attempt that only unwrapped the wrap on absolute positions
  didn't help, because the alive-masking of high-offset envs was the real culprit.)
- **What:** in `simulation/analysis/debug_agent.py` — added `_unwrap_clip_wrap()` and rewrote the
  x/y panels of `plot_root_tracking` to compute **per-env displacement from each env's own start**,
  unwrap the clip wrap, *then* average (`_mean_travel` helper). Now each env contributes only its
  few-metre travel, so deaths/wraps can't yank the mean. z (height) and orientation unchanged.
  Verified: ref forward travel 4.19 m vs policy 4.29 m at 3 s (speed ratio ~1.02), no negative
  jumps. Regenerated `…/5_best_try/07-58-27/Debug_data/root_tracking_07-58-27_20260620_174236.pdf`
  (same filename) and copied over `../thesis-doc/assets/` (same name).

### 2026-06-20 (later 7) — bigger titles/ticks on training-curve thesis figures

- **Why:** the wandb training-curve figures read too small in the thesis, and the policy-entropy
  panel had no title.
- **What:** in `final_document/scripts/plotters.py` — `plot_entropy_curve` now sets a title
  ("Policy entropy over training"); `plot_entropy_curve`, `plot_metric_for_sweep`, and
  `plot_wandb_metric_curve` (episode return/length) bump title→12.5 pt, axis labels→11 pt, tick
  labels→10 pt. Regenerated `training__policy_entropy__v1`, `…__metric_for_sweep__v1`,
  `…__episode_return__v1`, `…__episode_length__v1` into `final_document/images/` and copied the
  first three (the names the user named; episode_length is not in assets) over the same filenames
  in `../thesis-doc/assets/`.

### 2026-06-20 (later 6) — joint_angles plot now PDF + Times New Roman

- **Why:** match the rest of the thesis figures (vector + serif). `plot_joint_angles.py` wrote a
  raster PNG in the default sans font.
- **What:** added the same `_use_times_new_roman()` helper at module load and switched the output
  from `joint_angles_<stem>.png` to `.pdf` (dropped the now-irrelevant `dpi`). Regenerated
  `artifacts/figures/joint_angles_trajectory_adapted.pdf` (single gait cycle, embeds
  `TimesNewRomanPSMT`); old PNG removed.

### 2026-06-20 (later 5) — debug_agent figures now PDF + Times New Roman

- **Why:** thesis-quality figures want a vector format and a consistent serif face. The 7
  `debug_agent.py` plots were rasterised PNGs in matplotlib's default sans font.
- **What:** added `_use_times_new_roman()` at module load in `simulation/analysis/debug_agent.py`
  — explicitly registers the msttcorefonts `Times_New_Roman.ttf` (matplotlib's cache didn't index
  it) with metric-compatible fallbacks (Nimbus Roman / Liberation Serif), sets `font.family=serif`,
  `mathtext.fontset=stix`, `pdf.fonttype=42`. Switched all 7 output filenames from `.png` to
  `.pdf`. Regenerated the headline agent's set into
  `artifacts/trained_agents/curated/5_best_try/07-58-27/Debug_data/` (64 envs × 300 steps; old
  PNGs removed). Verified the PDFs embed `TimesNewRomanPSMT`.
- **NB:** `OUTPUT_DIR` defaults to the shared `paths.AGENT_DEBUG`; to write into an agent's own
  `Debug_data/` folder, override `debug_agent.OUTPUT_DIR` before calling `main()` (and add
  `simulation/analysis` to `sys.path` for the sibling `plot_joint_angles` import).

### 2026-06-20 (later 4) — standalone single-agent Cost-of-Transport figure

- **Why:** the existing `fig_cost_of_transport` draws a grouped *bar* chart, which is meaningless
  for the single headline agent (`5_best_try` / "trained agent").
- **What:** added `fig_cost_of_transport_single(records, label)` to
  `simulation/analysis/thesis_eval_figures.py` — plots the per-step mechanical-power trace
  `Σ_j|τ_j·q̇_j|` (the integrand of the CoT) over the deterministic rollout, with a mean-power
  dashed line and the resulting dimensionless `CoT = E/(m g d)` annotated. Wired into
  `plot_all`; output `artifacts/figures/thesis_eval/4_cost_of_transport_single.{pdf,png}`.
  For the headline agent: CoT ≈ 1.52, mean power ≈ 165 W, m = 7.83 kg.

### 2026-06-20 (later 3) — joint-angles limit bands from the canonical ROM spec table

- **Why:** `plot_joint_angles` drew its red limit bands from `senecabot_move.xml`, whose joint
  ranges differ from the actual robot spec (e.g. front hip [0,90]° in move vs the spec's
  [25,105]°), so the bands marked the wrong "actual" joint ranges.
- **What:** replaced the XML-read limits with a hardcoded `ROM_DEG` table quoted directly from the
  thesis design spec (front/back: hip 25..105 / −105..−40, knee 34..150 / −140..−40, ankle
  0..160 / −85..0). `model_joint_ranges()` now returns that table (radians); removed the
  `load_model`/`parameters`/`mujoco` XML dependency. Both callers (`plot_joint_angles` and
  `debug_agent`'s `ranges`) use it. Regenerated `artifacts/figures/joint_angles_trajectory_adapted.png`.
  (Files: `simulation/analysis/plot_joint_angles.py`.)

### 2026-06-20 (later 2) — joint_tracking debug plot: per-gait-cycle x-axis + tight y-axis

- **Why:** the `joint_tracking` debug figure plotted angle vs. *time* over the whole rollout with
  the y-axis stretched to the full joint-limit band, so the (small) policy-vs-reference tracking
  differences were illegible.
- **What:** `debug_agent.plot_joint_tracking` now folds the rollout into ONE representative gait
  cycle (x = % of gait cycle) via a new `_gait_phase_average` helper — cycles are segmented at the
  reference FR-hip troughs (same boundary as `augment.detect_cycles`; policy & reference share the
  env phase clock), each complete cycle resampled to 100 phase points and averaged across cycles,
  with a ±1σ policy band. Each panel's y-axis auto-scales to its own data range (the limit-band
  lines that forced the wide view are dropped). Falls back to the old time axis if <2 clean cycles
  are detected. Default `x_mode="gait"` (so normal `debug_agent.py` runs get the new view too).
- Regenerated for the trained agent to
  `artifacts/trained_agents/curated/5_best_try/07-58-27/Debug_data/joint_tracking_pct_gait.png`
  (new file; the original dated PNG was left untouched). (Files: `simulation/analysis/debug_agent.py`.)

### 2026-06-20 (later) — Thesis figures from the headline run's wandb export

- **Why:** the wandb history of the trained agent (`5_best_try` / `faithful-sunset-317`) was
  exported as per-metric CSVs into `final_document/data/Wandb_stats_5_best_try/` (15 validation
  distances + `entropy.csv`); the thesis needs vector figures from them in the house style.
- **What:** added two plotters to `final_document/scripts/plotters.py` —
  `plot_validation_distance_grid` (a 3×5 cluster: rows = Euclidean/DTW/Discrete-Fréchet, cols =
  Joint{Position,Velocity}/RelSite{Position,Orientation,Velocity}, each panel its own y-scale)
  and `plot_entropy_curve` — plus two registry entries
  (`error__validation_distances__grid`, `training__policy_entropy__v1`). Both read the wandb
  folder directly (`sources=()`) and key each series off the CSV's value-column header (canonical
  `…/<measure>/<quantity>`), so the few mislabelled filenames (`DTW_rrotvec`, `eucl_site_revl`)
  don't matter. Generated to `final_document/images/*.{pdf,png}` + manifest rows.
- **Recommendation captured:** the 15 distances are presented as ONE cluster (shared env-step
  x-axis, one "imitation improves over training" story); entropy is a standalone training-monitor
  figure. (Files: `final_document/scripts/plotters.py`, `final_document/scripts/registry.py`.)
- **Optuna figures:** `thesis_eval_figures.fig_optuna` now emits the slice plots
  (`3d_optuna_slice`) alongside history (`3b`) and fANOVA importance (`3c`) for study
  `ppo_phase2_rewards_v3` → `artifacts/figures/thesis_eval/`.
- **Follow-ups (same wandb folder):** added separate `training__episode_return__v1` and
  `training__episode_length__v1` curves (via a generic `plot_wandb_metric_curve`) and
  `plot_metric_for_sweep` (`training__metric_for_sweep__v1`, the validation Euclidean rel-site
  objective, sparse markers) from `MeanEpisode{Return,Length}.csv` + `Metric_for_sweep.csv`.
  The validation-distance grid also got 100 M-step x-ticks and per-quantity y-axis units
  (qpos/qvel flagged mixed: they include the free-root DoFs).

### 2026-06-20 — Thesis evaluation figures + single-cycle joint-angles plot

- **Why:** the thesis Agent-Evaluation section needs publication-quality (vector PDF) figures for
  the imitation-quality / locomotion-outcome / composite metrics, plus gait-quality plots, all
  measured from the curated agents.
- **New `simulation/analysis/thesis_eval_figures.py`** — computes, per curated agent, the 15
  framework validation distances (stochastic rollout through `MetricsHandler`, matching training
  validation), the `agent_diagnostics` scalars + composite, Cost of Transport, and a foot-contact
  gait-phase timeline (deterministic 64×300 rollout); then renders thesis-styled PDFs to
  `artifacts/figures/thesis_eval/` and pulls the headline run's training curves (entropy + the 15
  validation distances vs env-step) from the wandb API (run `faithful-sunset-317` = `5_best_try`).
  Records cached to `records.pkl` (`--plot-only` re-plots offline). **Only agents `4_best_try`
  (100M) and `5_best_try` (300M) are replayable** — `1/2/3_best_try` were trained against an older
  262-dim observation spec and the current fork builds 358-dim obs, so their normalization layer
  can't be reconstructed (skipped with a clear message). `5_best_try` is the headline agent
  (`BEST_LABEL`): speed_ratio 1.025, joint RMS 6.23°, composite 4.83, CoT 1.52 — beats 4_best_try
  on every metric.
- **`debug_agent.rollout` gained `collect_dynamics=False`** (default off → byte-identical for
  existing callers): when true it also records `qfrc_actuator` (for CoT) and per-foot `foot_z`
  (geometric stance detection for the gait diagram — MJX hides `data.contact` in this version, so
  contact is inferred from the foot-sphere height vs the floor).
- **`plot_joint_angles.py` now plots a single gait cycle by default** (trough-to-trough on the
  FR-hip angle `qpos[:,7]`, same detector as `augment.detect_cycles`; one cycle ≈ 127 frames ≈
  0.64 s) on a 0–100 % gait-cycle x-axis — far more legible than the full 60 s looped clip. Pass
  `--full` to restore the whole-clip view. (Files: `simulation/analysis/plot_joint_angles.py`,
  `simulation/analysis/debug_agent.py`, `simulation/analysis/thesis_eval_figures.py`.)

### 2026-06-18 — Added high-quality MuJoCo model renders (front/side/diagonal) to thesis figures

- **Why:** the thesis needs photoreal views of the training robot model. Added a standalone
  renderer `final_document/scripts/render_model.py` that loads
  `simulation/assets/xml/senecabot_loco.xml`, poses it in the project's **canonical standing
  stance** (front `[82,95,13]°`, rear `[-78,-120,-42]°`, base z=0.35 — taken verbatim from
  `notebooks/exploration/Body_display.ipynb`), drops the base so the lowest foot rests on z=0,
  hides all sites (mimic markers + foot sites) for a clean figure, and renders three free-camera
  views (front / side / diagonal) at 1920×1440.
- **Notes:** these are 3D renders, not CSV plots, so the script lives outside `generate_all.py`
  but reuses `io_utils` for the naming convention + manifest. Needs an on-screen GL backend —
  run with `MUJOCO_GL=glfw` (egl/osmesa are broken in this env). Renders go to
  `final_document/images/schematic__robot_model__{front,side,diagonal}.png` (category
  `schematic`) + manifest rows.
- **Files:** `final_document/scripts/render_model.py`; generated
  `final_document/images/schematic__robot_model__{front,side,diagonal}.png`.

### 2026-06-18 — Added project-overview architecture schematic to the thesis figure pipeline

- **Why:** the thesis needs one general diagram of the whole SenecaBot pipeline that also
  surfaces the author's local modifications to the `loco-mujoco-seneca` fork. Added plotter
  `plot_project_overview` (final_document/scripts/plotters.py, "schematic" category): a top-down
  block-flow — ovine C3D MOCAP → MOCAP→reference-trajectory pipeline (io/kinematics/build/cyclic/
  traj_gen, output `trajectory_adapted.npz` = root pose + 12 joint trajectories + `*_mimic` site
  targets) → robot model → imitation training (Hydra/PPO) → trained policy → evaluation. Fork-
  modified components (`MimicReward`, `PPOJax`) are starred ($\bigstar$, Wong orange) and a side
  panel lists the local changes (rootvel / track_root_xy / qvel_w_exp; live wandb logging +
  best-checkpoint; viewer `record_camera`), with a dashed connector into the training box.
- **Notes:** pure schematic, no CSV (sources=()). Stars use mathtext `$\bigstar$` (DejaVu Serif
  lacks U+2605). Registered as `schematic__project_overview__v1` in registry.py.
- **Files:** `final_document/scripts/plotters.py`, `final_document/scripts/registry.py`;
  generated `final_document/images/schematic__project_overview__v1.{pdf,png}` + manifest rows.

### 2026-06-17 — Documented `final_document/` and `hardware/` in this file (/init)

- **Why:** both directories exist and `final_document/` is an actively used subsystem (the
  thesis figure pipeline, referenced in several change-log entries), but neither appeared in the
  Architecture or Commands sections — so a fresh Claude instance wouldn't know how to build
  thesis figures or what `hardware/` is. Added a "Thesis figures" + "Hardware" subsection and
  the `generate_all.py` commands. Docs only; no code changed.
- **Files:** `CLAUDE.md`.

### 2026-06-15 — Added leg naming-convention schematic to the thesis figure pipeline

- **Why:** thesis needs a figure declaring the per-leg segment + joint-angle naming
  convention. Added plotter `plot_leg_convention` (final_document/scripts/plotters.py,
  "schematic" category) drawing one leg as a simple triple pendulum: a hatched base pivot,
  generic segment names (upper/lower/foot, with fore/hind anatomical equivalents), and large
  angle labels (θ_hip absolute; θ_knee, θ_ankle relative). Proportions use the real hindlimb
  lengths from senecabot_loco.xml. Registered as `schematic__leg_naming__v1` in registry.py.
- **Edit knobs:** geometry via registry kwargs (`seg_lengths`, `abs_angles_deg`, `figsize_in`)
  with no code change; labels/colors inside the plotter; global look in style.py.
- **Files:** `final_document/scripts/plotters.py`, `final_document/scripts/registry.py`;
  generated `final_document/images/schematic__leg_naming__v1.{pdf,png}` + manifest row.

### 2026-06-15 — Renamed leg-segment bodies in senecabot_loco.xml (generic, non-anatomical)

- **Why:** the old body names `*_thigh / *_shank / *_metatarsus` were hindlimb-specific and
  wrong on the front legs (a forelimb has a humerus/forearm/meta*carpus*, joints
  shoulder-elbow-carpus — not thigh/metatarsus). Since this is a simplified robot with four
  identical 3-DOF legs, no single anatomical scheme fits both ends; renamed to generic
  proximal→distal names: `*_upper / *_lower / *_foot` (12 bodies + their 12 `<contact><exclude>`
  refs). Added a header comment in the XML explaining the convention.
- **Left unchanged on purpose:** the joint names (`*_hip/knee/ankle_joint`) and the mimic SITE
  names (`*_thigh_mimic / *_shank_mimic / *_foot_mimic`) — both are referenced by name in
  `simulation/config/pipeline.py` (`SITE_NAMES`, `JOINT_NAMES_QPOS`) and the fork env
  `senecabot.py`, and are baked into the trained checkpoint's obs spec. So a body↔site word
  mismatch (body `fr_upper` holds site `fr_thigh_mimic`) is expected and documented in-file.
- **Scope:** training model only. The other XMLs (`senecabot_move.xml`, `senecabot.xml`,
  `pata_trasera_bonilla*.xml`) still use the old body names; they are independent, non-training
  models and were left as-is.
- **Files:** `simulation/assets/xml/senecabot_loco.xml`. No Python touched (body names are not
  referenced in code). Verified: model recompiles (14 bodies / 13 joints) and all
  `SITE_NAMES` + `JOINT_NAMES_QPOS` still resolve.

### 2026-06-12 — CLAUDE.md finalized

- Project complete; rewrote this file as a final-state reference: removed resolved/retracted
  Known Issues, the diagnostics-in-progress narratives, and the full running change log
  (condensed into "Project history" above; full text in git history). No code changed.

<!-- New entries go ABOVE this line, newest first. -->
