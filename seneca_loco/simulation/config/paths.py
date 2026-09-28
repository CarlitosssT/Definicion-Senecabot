"""paths.py — single source of truth for every filesystem path in the project.

Import this instead of recomputing Path(__file__).parent.parent.parent. Inputs
(data, assets, config) live under the repo; ALL generated output lives under
artifacts/ (gitignored). Override the repo root with $SENECA_HOME if needed
(keep it in sync with conf.yaml's hydra.run.dir, which uses the same env var).

pipeline.py stays the home of the mocap *constants* (rates, marker map, layout)
and keeps exporting its own path constants for the mocap package; this module
is the superset the rest of the code should import.
"""
import os
from pathlib import Path

# repo root: .../seneca_loco/simulation/config/paths.py -> parents[2] = seneca_loco
PROJECT_ROOT = Path(os.environ.get("SENECA_HOME", Path(__file__).resolve().parents[2]))

# --- inputs (tracked) ---
DATA_DIR     = PROJECT_ROOT / "data"
C3D_DIR      = DATA_DIR / "C3D_Final"
PROC_DIR     = DATA_DIR / "processed_data"
CONF_DIR     = PROJECT_ROOT / "simulation" / "training"   # where the Hydra conf.yaml lives
XML_DIR      = PROJECT_ROOT / "simulation" / "assets" / "xml"
XML_PATH     = XML_DIR / "senecabot_loco.xml"
TRAJ_ADAPTED = PROC_DIR / "trajectory_adapted.npz"

# --- generated outputs (gitignored) ---
ARTIFACTS      = PROJECT_ROOT / "artifacts"
TRAINED_AGENTS = ARTIFACTS / "trained_agents"
OPTUNA_DIR     = ARTIFACTS / "optuna"
WANDB_DIR      = ARTIFACTS / "wandb"
RECORDINGS     = ARTIFACTS / "recordings"
FIGURES        = ARTIFACTS / "figures"          # replaces outputs_project/
AGENT_DEBUG    = FIGURES / "agent_debug"

for _d in (ARTIFACTS, TRAINED_AGENTS, OPTUNA_DIR, WANDB_DIR, RECORDINGS, FIGURES, AGENT_DEBUG):
    _d.mkdir(parents=True, exist_ok=True)
