"""
pipeline.py

Single source of truth for the MOCAP -> trajectory pipeline.

Everything that used to be a magic number scattered across the notebooks and
``trajectory_generation.py`` lives here: the true capture frequency, the project
paths, the marker -> joint-angle map (with its per-leg sign conventions), the
root-marker offset, and the qpos/site layout expected by Loco-MuJoCo.

The capture frequency is the linchpin of the time-consistency work
(see ``TODO_time_consistency_check.md``):

    POINT_RATE  ->  qvel dt in build.py  ->  frequency stored in the .npz
                ->  traj_info.frequency  ->  Loco-MuJoCo interpolation to control_dt

Keeping it in one place means the whole chain stays truthful.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Capture frequency
# ---------------------------------------------------------------------------
# The ovine C3D files were captured with a POINT (marker) rate of 200 Hz.
# NOTE: the 2000 Hz figure that used to be hardcoded is the ANALOG / force-plate
# rate, a different channel -- NOT the kinematic sample rate. io.load_c3d reads
# the real value from the header; this constant is the fallback / sanity check.
POINT_RATE = 200.0  # Hz

# Cycles are resampled to this many frames during augmentation.
CYCLE_FRAMES = 100

# Env control rate for SenecaBot: env.dt = timestep(0.001) * n_substeps(10) = 0.01 s.
# Used ONLY for the frequency-consistency sanity print in trajectory_generation.py;
# the stored trajectory frequency is always the capture rate (POINT_RATE), and
# Loco-MuJoCo interpolates capture_rate -> control_rate at load time. Keep in sync
# with the env config (conf.yaml / senecabot.py); it is informational, not load-bearing.
CONTROL_RATE = 100.0  # Hz

# ---------------------------------------------------------------------------
# Project paths (resolved from this file, OS-independent)
# ---------------------------------------------------------------------------
#   .../seneca_loco/simulation/config/pipeline.py
#   parents[0]=config  parents[1]=simulation  parents[2]=seneca_loco
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR    = PROJECT_ROOT / "data"
C3D_DIR     = DATA_DIR / "C3D_Final"
PROC_DIR    = DATA_DIR / "processed_data"
OUTPUTS_DIR = PROJECT_ROOT / "artifacts" / "figures"   # kept in sync with paths.FIGURES
XML_PATH    = PROJECT_ROOT / "simulation" / "assets" / "xml" / "senecabot_loco.xml"

# Standard artifact names produced along the chain.
CSV_NAME       = "ovino_angles.csv"
TRAJ_NAME      = "trajectory.npz"
AUGMENTED_NAME = "trajectory_augmented.npz"
ADAPTED_NAME   = "trajectory_adapted"  # Trajectory.save appends .npz

# ---------------------------------------------------------------------------
# Root body (torso) extraction
# ---------------------------------------------------------------------------
# The torso pose is taken from the T13 vertebra marker. Mocap is in millimetres;
# the Z offset re-zeros the vertical so the robot stands at a sensible height.
ROOT_MARKER     = "T13"
MM_TO_M         = 1.0 / 1000.0
T13_Z_OFFSET_MM = 350.0

# ---------------------------------------------------------------------------
# Markers extracted from each C3D file
# ---------------------------------------------------------------------------
MARKERS = [
    "L_GTUB", "L_LEPIRAD", "L_METAR", "LF_DPHAL",   # front left
    "R_GTUB", "R_LEPIRAD", "R_METAR", "RF_DPHAL",   # front right
    "L_GTROC", "L_LEPI", "L_LMAL", "LH_PPHAL",      # back left
    "R_GTROC", "R_LEPI", "R_LMAL", "RH_PPHAL",      # back right
    ROOT_MARKER,
]

# ---------------------------------------------------------------------------
# Marker -> joint-angle map
# ---------------------------------------------------------------------------
# Each leg has a 3-segment chain. Angles are computed in the sagittal (Y-Z)
# plane with calc_joint_angle(A, B, s1, s2) = atan2(s2*(B.y-A.y), s1*(B.z-A.z)).
#
#   hip   : absolute   angle of segment A->B
#   knee  : relative to hip   = calc(A->B) - hip
#   ankle : relative chain    = calc(A->B) + ankle_hip*hip + ankle_knee*knee
#
# The per-leg signs and the ankle combination coefficients are empirical
# (front vs back legs differ). They were verified against the skeleton
# animation; preserved here exactly as the original notebook computed them.
#
# Historical note: the middle segment was called "shoulder" in the old CSV;
# it drives the robot's *knee* joint, so it is named "knee" here to remove the
# long-standing shoulder/knee naming trap.
#
# Format:
#   hip   : (marker_A, marker_B, sign_z, sign_y)
#   knee  : (marker_A, marker_B, sign_z, sign_y)
#   ankle : (marker_A, marker_B, sign_z, sign_y, hip_coeff, knee_coeff)
LEG_CHAINS = {
    "fl": {
        "hip":   ("L_GTUB", "L_LEPIRAD", -1, -1),
        "knee":  ("L_LEPIRAD", "L_METAR", 1, 1),
        "ankle": ("L_METAR", "LF_DPHAL", 1, -1, +1, -1),
    },
    "fr": {
        "hip":   ("R_GTUB", "R_LEPIRAD", -1, -1),
        "knee":  ("R_LEPIRAD", "R_METAR", 1, 1),
        "ankle": ("R_METAR", "RF_DPHAL", 1, -1, +1, -1),
    },
    "bl": {
        "hip":   ("L_GTROC", "L_LEPI", -1, -1),
        "knee":  ("L_LEPI", "L_LMAL", 1, 1),
        "ankle": ("L_LMAL", "LH_PPHAL", -1, 1, -1, +1),
    },
    "br": {
        "hip":   ("R_GTROC", "R_LEPI", -1, -1),
        "knee":  ("R_LEPI", "R_LMAL", 1, 1),
        "ankle": ("R_LMAL", "RH_PPHAL", -1, 1, -1, +1),
    },
}

# Order of legs/joints as written into the angle DataFrame and consumed by build.
LEG_ORDER   = ["fl", "fr", "bl", "br"]
JOINT_ORDER = ["hip", "knee", "ankle"]

def angle_columns():
    """Column names of the joint angles, e.g. 'fl_hip_angle'."""
    return [f"{leg}_{j}_angle" for leg in LEG_ORDER for j in JOINT_ORDER]

# ---------------------------------------------------------------------------
# Loco-MuJoCo qpos / site layout (consumed by build.py and trajectory_generation)
# ---------------------------------------------------------------------------
# qpos = [root(7)] + 12 hinge joints, in this exact order.
JOINT_NAMES_QPOS = [
    "root",
    "fr_hip_joint", "fr_knee_joint", "fr_ankle_joint",
    "fl_hip_joint", "fl_knee_joint", "fl_ankle_joint",
    "br_hip_joint", "br_knee_joint", "br_ankle_joint",
    "bl_hip_joint", "bl_knee_joint", "bl_ankle_joint",
]

# Sites used by the mimic reward (must match senecabot_loco.xml).
SITE_NAMES = [
    "fr_foot_mimic", "fl_foot_mimic", "br_foot_mimic", "bl_foot_mimic", "base_mimic",
    "fr_shank_mimic", "fl_shank_mimic", "br_shank_mimic", "bl_shank_mimic",
]
