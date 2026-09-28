"""
Initial geometry and masses of the robot, as reported in Nicolás' thesis (`tesis nicolas.pdf`).

The values are stored here so the pipeline does not need the PDF. They were transcribed from:
  * Table V (p. 32)            - segment length [cm] (front / back legs), enforced range of motion [deg]
                                 and actuator torque limit [N·m] of the hip, knee and ankle joints.
  * Sec. 3.2.1 (pp. 31-32)     - "The only body assigned an explicit mass is the torso (3 kg); the mass
                                 and full inertia tensor of every leg link are instead computed
                                 automatically by MuJoCo from the capsule geometry under a uniform density."
  * Table II (p. 11)           - goat hindlimb segment masses/lengths (biological reference only; the
                                 thesis states they informed the segment lengths, not the model's masses).

`check_against_model` cross-checks them against the MuJoCo XML the policies were trained on.
"""
import mujoco
import numpy as np

from . import config as C

JOINTS = ("hip", "knee", "ankle")
SEG_OF_JOINT = {"hip": "upper", "knee": "lower", "ankle": "foot"}   # body that each joint drives
LEGS = ("fr", "fl", "br", "bl")

# Table V (p. 32): segment length [m]
SEGMENT_LENGTH_M = {
    ("front", "hip"): 0.121, ("back", "hip"): 0.166,
    ("front", "knee"): 0.111, ("back", "knee"): 0.231,
    ("front", "ankle"): 0.153, ("back", "ankle"): 0.148,
}
# Table V (p. 32): range of motion [deg] (back legs mirrored, sign-flipped)
ROM_DEG = {
    ("front", "hip"): (25.0, 105.0), ("back", "hip"): (-105.0, -40.0),
    ("front", "knee"): (34.0, 150.0), ("back", "knee"): (-140.0, -40.0),
    ("front", "ankle"): (0.0, 160.0), ("back", "ankle"): (-85.0, 0.0),
}
# Table V (p. 32): actuator torque limit [N·m] (symmetric, ±)
TORQUE_LIMIT_NM = {"hip": 80.0, "knee": 80.0, "ankle": 40.0}
# Sec. 3.2.1 (pp. 31-32): the only explicit mass of the model
TORSO_MASS_KG = 3.0
# Table II (p. 11), adapted from [21]: goat hindlimb segments (reference only, not used by the model)
GOAT_HINDLIMB = {
    "thigh": dict(mass_kg=1.680, length_m=0.166),
    "shank": dict(mass_kg=0.795, length_m=0.231),
    "foot": dict(mass_kg=0.099, length_m=0.148),
    "toes": dict(mass_kg=0.081, length_m=0.085),
}
SOURCE = "Nicolás' thesis (tesis nicolas.pdf): Table V p. 32, Sec. 3.2.1 pp. 31-32, Table II p. 11"


def load():
    """Thesis parameters: lengths [m], ROM [deg], torque limits [N·m], torso mass [kg]."""
    return dict(segment_length_m=dict(SEGMENT_LENGTH_M), rom_deg=dict(ROM_DEG),
                torque_limit_Nm=dict(TORQUE_LIMIT_NM), torso_mass_kg=TORSO_MASS_KG,
                goat_hindlimb={k: dict(v) for k, v in GOAT_HINDLIMB.items()}, source=SOURCE)


def check_against_model(params, xml=C.XML_PATH, tol_m=0.002):
    """Compare the thesis values with the MuJoCo model. Returns (ok, human-readable lines)."""
    model = mujoco.MjModel.from_xml_path(str(xml))
    lines, ok = [], True
    for leg in LEGS:
        side = "front" if leg[0] == "f" else "back"
        for j in JOINTS:
            b = model.body(f"{leg}_{SEG_OF_JOINT[j]}").id
            caps = [g for g in range(model.ngeom)
                    if model.geom_bodyid[g] == b and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]
            L_xml = 2 * float(model.geom_size[caps[0], 1])          # capsule half-length -> length
            L_thesis = params["segment_length_m"][(side, j)]
            if abs(L_xml - L_thesis) > tol_m:
                ok = False
                lines.append(f"  MISMATCH {leg}_{j}: thesis {L_thesis*100:.1f} cm vs XML {L_xml*100:.2f} cm")
            jid = model.joint(f"{leg}_{j}_joint").id
            r_xml = tuple(np.degrees(model.jnt_range[jid]).round(1))
            r_thesis = params["rom_deg"][(side, j)]
            if not np.allclose(r_xml, r_thesis, atol=0.6):
                ok = False
                lines.append(f"  MISMATCH {leg}_{j} ROM: thesis {r_thesis} vs XML {r_xml}")
    torso_xml = float(model.body_mass[model.body("base").id])
    if abs(torso_xml - params["torso_mass_kg"]) > 1e-6:
        ok = False
        lines.append(f"  MISMATCH torso mass: thesis {params['torso_mass_kg']} kg vs XML {torso_xml} kg")
    for i in range(model.nu):
        j = model.actuator(i).name.split("_")[1]
        if abs(model.actuator_ctrlrange[i, 1] - params["torque_limit_Nm"][j]) > 1e-6:
            ok = False
            lines.append(f"  MISMATCH {model.actuator(i).name} torque limit")
    lines.insert(0, "thesis <-> XML: all segment lengths, ROMs, torso mass and torque limits match"
                 if ok else "thesis <-> XML: discrepancies found")
    return ok, lines
