"""
Actuator selection: for every joint, the LIGHTEST catalog motor that satisfies

    peak_torque    >= FS_TORQUE * |tau|max            (always)
    nominal_torque >= FS_TORQUE * tau_rms              (CHECK_RMS_TORQUE: continuous / thermal)
    rated speed    >= FS_SPEED  * |omega|max           (CHECK_SPEED: the gait must be reachable)

Ties in mass are broken by the larger peak torque. Joints may get different motors; with
SYMMETRIC_MOTORS the left and right joint of each pair (FR/FL, BR/BL) are sized with the larger
requirement of the two and therefore get the same motor.
If no motor qualifies the strongest one is returned and flagged `feasible=False`.
"""
import numpy as np

from . import config as C
from .model_builder import JOINT_NAMES


_MIRROR = {"fr": "fl", "fl": "fr", "br": "bl", "bl": "br"}


def _requirements(summary):
    tau_max, tau_rms, w_max = (np.asarray(summary[k], dtype=float) for k in ("tau_max", "tau_rms", "omega_max_rpm"))
    if C.SYMMETRIC_MOTORS:
        mirror = np.array([JOINT_NAMES.index(_MIRROR[n[:2]] + n[2:]) for n in JOINT_NAMES])
        tau_max, tau_rms, w_max = (np.maximum(x, x[mirror]) for x in (tau_max, tau_rms, w_max))
    return tau_max, tau_rms, w_max


def select(catalog, summary):
    out = []
    tau_max, tau_rms, w_max = _requirements(summary)
    for j, name in enumerate(JOINT_NAMES):
        req = dict(peak=C.FS_TORQUE * tau_max[j],
                   nominal=C.FS_TORQUE * tau_rms[j] if C.CHECK_RMS_TORQUE else 0.0,
                   speed=C.FS_SPEED * w_max[j] if C.CHECK_SPEED else 0.0)
        ok = ((catalog["peak_torque_Nm"] >= req["peak"]) & (catalog["nominal_torque_Nm"] >= req["nominal"])
              & (catalog["speed_rpm"] >= req["speed"]))
        feasible = bool(ok.any())
        pool = catalog[ok] if feasible else catalog
        order = ["motor_mass_kg", "peak_torque_Nm"] if feasible else ["peak_torque_Nm", "motor_mass_kg"]
        asc = [True, False] if feasible else [False, True]
        best = pool.sort_values(order, ascending=asc).iloc[0]
        margins = {"peak torque": best["peak_torque_Nm"] / max(req["peak"], 1e-9),
                   "RMS torque": best["nominal_torque_Nm"] / max(req["nominal"], 1e-9) if req["nominal"] else np.inf,
                   "speed": best["speed_rpm"] / max(req["speed"], 1e-9) if req["speed"] else np.inf}
        out.append(dict(
            joint=name, model=best["model"], mass=float(best["motor_mass_kg"]),
            peak=float(best["peak_torque_Nm"]), nominal=float(best["nominal_torque_Nm"]),
            speed=float(best["speed_rpm"]), feasible=feasible, binding=min(margins, key=margins.get),
            req_peak=req["peak"], req_nominal=req["nominal"], req_speed=req["speed"],
        ))
    return out
