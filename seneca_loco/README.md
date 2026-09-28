# 🐑 SenecaBot · `seneca_loco`

> Imitation learning for a quadruped robot (**SenecaBot**) to reproduce an ovine
> (sheep) gait from motion-capture (MOCAP) data, using *DeepMimic*-style imitation
> on top of [LocoMuJoCo](https://github.com/robfiras/loco-mujoco).

**Author:** Nicolás Andrade Manrique — `20221204`
**Context:** Undergraduate thesis / individual project, Universidad de los Andes.

---

## 📚 Table of contents

- [Overview](#-overview)
- [Tech stack](#-tech-stack)
- [Installation](#-installation)
- [Quick start](#-quick-start)
- [MOCAP → trajectory data pipeline](#-mocap--trajectory-data-pipeline)
- [Training and evaluation](#-training-and-evaluation)
- [Project structure](#-project-structure)
- [MuJoCo models](#-mujoco-models)
- [Notes and conventions](#-notes-and-conventions)

---

## 🎯 Overview

The goal is to train **SenecaBot** (free root + 12 hinge joints) to imitate a
reference gait derived from ovine motion capture. The full flow goes from the raw
`.c3d` files to a trained PPO agent:

```
C3D (MOCAP)  →  joint angles  →  qpos/qvel  →  LocoMuJoCo trajectory  →  PPO
```

All processing logic lives in the `simulation/` package (not in the notebooks),
with a **single source of truth** for frequencies, paths and conventions in
[`simulation/config/pipeline.py`](simulation/config/pipeline.py).

---

## 🧰 Tech stack

| Component | Use |
|---|---|
| **LocoMuJoCo** | Imitation benchmark (installed editable from the local fork `loco-mujoco-seneca`) |
| **MuJoCo / MJX / MjWarp** | Physics simulation (CPU and GPU via JAX) |
| **JAX + Flax** | Accelerated compute and neural networks |
| **PPO (`PPOJax`)** | RL algorithm |
| **Hydra + OmegaConf** | Configuration management (`conf.yaml`) |
| **Weights & Biases** | Experiment logging |
| **Optuna** | Hyperparameter optimization |
| **ezc3d / pandas / scipy** | MOCAP reading and processing |

> **Recommended environment:** conda env `workspace` (with `loco_mujoco`, `jax`, `mujoco`).

---

## 🚀 Installation

### Method 1 — Direct install

```bash
git clone https://github.com/Robiolab/seneca_loco.git
cd seneca_loco
pip install -e .
```

### Method 2 — With the LocoMuJoCo fork (recommended)

`seneca_loco` depends on a fork of LocoMuJoCo. To work on both in editable mode:

```bash
# Step 1. Create a container folder (in my case: workspace)
mkdir workspace && cd workspace

# Step 2. Clone both repositories
git clone https://github.com/Robiolab/seneca_loco.git
git clone https://github.com/<user>/loco-mujoco-seneca.git   # loco-mujoco fork

# Step 3. Install both in editable mode
cd seneca_loco        && pip install -e .
cd ../loco-mujoco-seneca && pip install -e .
```

Optional GPU support:

```bash
pip install jax["cuda12"]
```

---

## ⚡ Quick start

```bash
# Interactive launcher (menu over all the main tasks below)
python main.py

# Walking animation
python scripts/main.py

# Full MOCAP → trajectory pipeline (see next section)
python -m simulation.mocap.run --augment
```

---

## 🔬 MOCAP → trajectory data pipeline

Processing is modularized in `simulation/mocap/`. Each stage is an importable
function and everything is orchestrated by a single CLI.

| Stage | Module | Input → Output |
|---|---|---|
| 1. C3D reading | `mocap/io.py` | `.c3d` → marker DataFrame + real frequency |
| 2. Kinematics | `mocap/kinematics.py` | markers → joint angles |
| 3. Build | `mocap/build.py` | angles → `qpos`/`qvel` (MuJoCo FK) → `trajectory.npz` |
| 4. Augmentation | `mocap/augment.py` | `trajectory.npz` → `trajectory_augmented.npz` |
| 5. Adaptation | `mocap/trajectory/trajectory_generation.py` | `*.npz` → `trajectory_adapted.npz` (LocoMuJoCo) |

**Run everything (stages 1–4) with a time-consistency report:**

```bash
python -m simulation.mocap.run [--augment] [--file-index N] [--c3d PATH.c3d]
```

Then generate the final training trajectory (stage 5). By default it uses the base
`trajectory.npz`; the augmented dataset is **strictly opt-in** (never auto-detected):

```bash
python -m simulation.mocap.trajectory.trajectory_generation                     # uses trajectory.npz
python -m simulation.mocap.trajectory.trajectory_generation +use_augmented=true # uses trajectory_augmented.npz
```

> ⏱️ **Frequency consistency:** the real capture rate is **200 Hz** (the C3D *POINT*
> rate; the 2000 Hz figure is the analog / force-plate rate). The CLI verifies that
> `capture frequency → qvel dt → stored frequency` all agree, so that LocoMuJoCo
> interpolates correctly to the environment's control rate.

---

## 🏋️ Training and evaluation

From the repo root (conda env `workspace`); all generated output lands in `artifacts/`:

```bash
# Train (Hydra, reads simulation/training/conf.yaml; saves to artifacts/trained_agents/)
python simulation/training/train.py

# Replay a trained agent
python simulation/training/eval.py --path artifacts/trained_agents/<date>/<time>/PPOJax_saved.pkl [--use_mujoco]

# Agent diagnostics (figures in artifacts/figures/agent_debug/)
python simulation/analysis/debug_agent.py [--path .../PPOJax_saved.pkl] [--n_envs 64] [--n_steps 300] [--stochastic]

# Scalar diagnostics + composite (Optuna Phase-2 objective)
python simulation/analysis/agent_diagnostics.py --path .../PPOJax_saved.pkl

# Trajectory joint angles vs the model's joint limits
python simulation/analysis/plot_joint_angles.py [path/to/trajectory.npz]
```

Hyperparameter optimization: `notebooks/optuna_optimization.ipynb` (2 phases: PPO and reward weights).

---

## 📁 Project structure

```
seneca_loco/
├── data/                          # INPUT data only
│   ├── C3D_Final/                 # Raw MOCAP captures (.c3d)
│   └── processed_data/            # ovino_angles.csv, trajectory*.npz, trajectory_adapted.npz
├── artifacts/                     # ALL generated output (gitignored)
│   ├── trained_agents/            # Hydra run dirs + curated/ (hand-picked best runs)
│   ├── figures/                   # videos, plots, agent_debug/ diagnostics
│   ├── optuna/                    # study DBs + per-trial run dirs
│   ├── wandb/  recordings/
├── simulation/                    # importable package
│   ├── assets/xml/                # MuJoCo models (see below)
│   ├── config/
│   │   ├── paths.py               # ⭐ Single source for ALL filesystem paths ($SENECA_HOME)
│   │   ├── pipeline.py            # ⭐ Single source: frequency, marker→joint map, layout
│   │   └── parameters.py          # Robot geometric parameters (templated XML)
│   ├── core/
│   │   ├── data_loader.py         # load_model (with XML includes), interpolate, resample_to_n
│   │   └── utils.py               # Image/video rendering, frame extraction
│   ├── mocap/                     # ⭐ MOCAP → trajectory pipeline
│   │   ├── io.py                  # C3D reading
│   │   ├── kinematics.py          # Markers → angles
│   │   ├── build.py               # Angles → qpos/qvel (FK)
│   │   ├── augment.py             # Data augmentation (phase/mirror/time/noise)
│   │   ├── cyclic.py              # Gait-cycle extraction, symmetrization, looping
│   │   ├── run.py                 # Pipeline CLI + consistency report
│   │   └── trajectory/            # Stage 5: trajectory_generation.py, trajectory_playback.py
│   ├── movement/walking.py        # Walking animation (scripts/main.py)
│   ├── analysis/                  # debug_agent.py, agent_diagnostics.py, plot_joint_angles.py
│   └── training/                  # train.py, eval.py, conf.yaml (Hydra config stays here)
├── notebooks/
│   ├── optuna_optimization.ipynb  # 2-phase HPO
│   └── exploration/               # visualization-only notebooks
├── hardware/                      # Arduino potentiometer + live sensing viewer
├── scripts/                       # main.py (scripted walk), image_extractor.py
├── final_document/                # Thesis figure-generation scripts
├── docs/
├── requirements.txt
├── pyproject.toml
└── README.md
```

> The exploration notebooks are **visualization-only**: they import the logic from
> `simulation.mocap` / `simulation.core`, with no duplicated code.

---

## 🤖 MuJoCo models (`simulation/assets/xml/`)

| File | Role |
|---|---|
| `senecabot_loco.xml` | **Training** model (free root + 12 hinges, `*_mimic` sites, includes `scene.xml`/`mimic_sites.xml`) |
| `senecabot_move.xml` | Templated model for movement generation (joints `fr_hip`…, all limited) |
| `senecabot.xml` | Older templated model (positive-only ranges) |
| `scene.xml`, `mimic_sites.xml` | Scene and reward sites (included by the above) |

**`qpos` layout:** `[root_pos(3), root_quat(4), 12 joints]`, in the order
`fr_hip, fr_knee, fr_ankle, fl_*, br_*, bl_*`.

---

## 📝 Notes and conventions

- **Capture frequency:** 200 Hz (the C3D POINT rate). Defined in `config/pipeline.py`.
- **Joint naming:** each leg's middle segment drives the robot's *knee* joint
  (previously labeled *shoulder* in the old CSVs).
- **Environment control rate:** `1/env.dt` (currently 100 Hz); LocoMuJoCo interpolates
  the trajectory from 200 → 100 Hz when loading it.
- **Reproducibility:** augmentation uses fixed seeds (`generate_augmented(..., seed=42)`).
- See [`CLAUDE.md`](CLAUDE.md) for the change log, known issues and context for
  agents/collaborators, and [`TODO_time_consistency_check.md`](TODO_time_consistency_check.md)
  for the time-consistency verification.
