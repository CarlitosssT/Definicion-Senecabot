"""
Central configuration of the geometry-optimization pipeline.

Everything a user is expected to tune lives here: safety factors, convergence, motor-selection
criteria, material properties and the trained agents to size the robot for. Units are SI inside
the code (m, kg, N, Pa); the report converts to mm / MPa.
"""
from pathlib import Path

# ------------------------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent                      # .../Optimizacion Geometria/geometry_opt
WORKSPACE = ROOT.parent                                     # .../Optimizacion Geometria
SENECA = WORKSPACE / "seneca_loco"
XML_PATH = SENECA / "simulation" / "assets" / "xml" / "senecabot_loco.xml"
CATALOG_CSV = WORKSPACE / "my actuator" / "motors_MyActuator_Series_L_H_X.csv"
RESULTS_DIR = ROOT / "results"                              # cached gaits (+ outputs of the default mode)
SUMMARY_MD = ROOT / "results_summary.md"

# The two trained policies to size the robot for, each with the reference it was trained on.
MODELS = {
    "v0218": dict(
        label="100M @ 0.218 m/s (no-slip reference)",
        agent=SENECA / "artifacts/trained_agents/2026-09-27/07-12-14/PPOJax_saved.pkl",
        reference=SENECA / "data/processed_data/trajectory_adapted.noslip_0218.npz",
    ),
    "v140": dict(
        label="100M @ 1.40 m/s (mocap-speed reference)",
        agent=SENECA / "artifacts/trained_agents/2026-09-27/18-22-14/PPOJax_saved.pkl",
        reference=SENECA / "data/processed_data/trajectory_adapted.preRootSpeed.bak.npz",
    ),
}

# ------------------------------------------------------------------------------------------------
# Design parameters and safety factors
# ------------------------------------------------------------------------------------------------
FS_TORQUE = 1.2            # motor: peak_torque >= FS_TORQUE * |tau|max   (and nominal >= FS_TORQUE * tau_rms)
FS_MATERIAL = 1.2          # structure: sigma_adm = sigma_yield / FS_MATERIAL
FS_SPEED = 1.0             # motor: rated max speed >= FS_SPEED * |omega|max
MIN_WALL_THICKNESS_MM = 4.0

# Same motor on the left and right leg of each joint (FR=FL, BR=BL): the pair is sized with the larger
# requirement of the two sides. False = every joint independently (the policies' gaits are not exactly
# left/right symmetric, so mirrored joints can land on different motors).
SYMMETRIC_MOTORS = False

# Motor-selection criteria (peak torque is always enforced)
CHECK_RMS_TORQUE = True    # continuous (thermal) torque: nominal_torque >= FS_TORQUE * tau_rms
CHECK_SPEED = True         # the joint speeds of the gait must be reachable by the motor

# Convergence of the dynamics <-> motor-selection loop
MASS_TOL = 0.01            # relative total-mass change between iterations
MAX_ITER = 10
CONVERGENCE_MODE = "all"   # "all": motors unchanged AND mass within tol;  "any": either one

# Joint damping of the thesis model (1.5 N·m·s/rad hip/knee, 0.05 ankle) is a stability
# regularizer of the simulation, not an actuator property; the motor pays damping*omega, which at
# the knee's peak speed is a large share of its torque. None = keep the thesis value (conservative,
# consistent with the simulated gait); a number [N·m·s/rad] replaces it on every leg joint.
JOINT_DAMPING_OVERRIDE = None

# ------------------------------------------------------------------------------------------------
# Structure model
# ------------------------------------------------------------------------------------------------
# Each leg link is a hollow circular tube (wall = MIN_WALL_THICKNESS_MM) of the segment's length.
# From iteration 1 on, the link's mass in the dynamic model is the tube's mass (the thesis model's
# solid capsules at water density are only used at iteration 0).
STRUCTURE_MASS_IN_LOOP = True
FOOT_SPHERE_INFILL = 1.0   # the r=25 mm contact sphere is printed in the link material (1.0 = solid)
MIN_OUTER_RADIUS_MM = MIN_WALL_THICKNESS_MM   # smallest tube = solid rod of diameter 2*wall
MAX_OUTER_RADIUS_MM = 150.0                   # search bound for the sizing solver

# ------------------------------------------------------------------------------------------------
# Materials (FDM 3D printing). Values of 3D-PRINTED specimens from the manufacturer TDS.
# ------------------------------------------------------------------------------------------------
# Prusament TDS (Prusa Polymers, rev. 2021-10), ISO 527-1 / ISO 1183:
#   PLA : tensile yield 51 +/- 3 MPa printed horizontal (59 +/- 2 vertical-xz, 57 filament),
#         tensile modulus 2.3 GPa, flexural strength 83 MPa, density 1.24 g/cm3
#   PETG: tensile yield 47 +/- 2 MPa printed horizontal (50 +/- 1 vertical-xz, 46 filament),
#         tensile modulus 1.5 GPa, flexural strength 66 MPa, density 1.27 g/cm3
# The lower (horizontal) printed value is used. Neither TDS reports strength ACROSS layers (z);
# if a tube is printed standing (layers perpendicular to its axis) set ANISOTROPY_FACTOR to
# ~0.5-0.7 (typical inter-layer strength retention reported in the FDM literature).
ANISOTROPY_FACTOR = 1.0
MATERIALS = {
    "PLA": dict(
        sigma_yield_MPa=51.0, E_GPa=2.3, density_kg_m3=1240.0,
        source="Prusament PLA TDS 2021-10: printed horizontal, ISO 527-1",
    ),
    "PETG": dict(
        sigma_yield_MPa=47.0, E_GPa=1.5, density_kg_m3=1270.0,
        source="Prusament PETG TDS 2021-10: printed horizontal, ISO 527-1",
    ),
}

# ------------------------------------------------------------------------------------------------
# Gait recording (trained policy, original thesis model) and inverse dynamics
# ------------------------------------------------------------------------------------------------
GAIT_N_ENVS = 16           # parallel rollouts of the deterministic policy
GAIT_STEPS = 900           # control steps (0.01 s); < horizon so a full run is not flagged done
GAIT_WARMUP = 50           # discard the first control steps (reset transient)
GAIT_SEED = 0

# Contact-force redistribution when masses change (per time sample, NNLS on friction-pyramid edges):
#   min  W^2 ||A f - b||^2 + LAMBDA ||f - f_recorded||^2 ,  f inside the friction pyramid
# A f = b are the 6 floating-base equations (the unactuated root must be balanced by the ground).
ROOT_EQ_WEIGHT = 1.0e3
CONTACT_REG_LAMBDA = 1.0
# The trained gait leans on the joint-limit stops (hind ankles up to ~24 N·m, hind hips ~9 N·m).
# True: the mechanical stops carry that load (as in the simulation). False: the motor must supply it.
LIMITS_CARRY_LOAD = True


def output_dir():
    """Where a run writes its outputs: results/ (default) or results/symmetric_motors/."""
    return RESULTS_DIR / "symmetric_motors" if SYMMETRIC_MOTORS else RESULTS_DIR


def summary_path():
    return ROOT / ("results_summary_symmetric.md" if SYMMETRIC_MOTORS else "results_summary.md")
