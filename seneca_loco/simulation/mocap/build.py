"""
build.py

Convert a joint-angle DataFrame into a Loco-MuJoCo-style qpos/qvel trajectory by
driving the MuJoCo model frame-by-frame and reading back generalized positions.

This is the logic previously embedded in Body_display.ipynb, with two fixes:
  * the finite-difference dt is 1 / capture_rate (a real, single-sourced number),
  * the knee joints are driven by the '*_knee_angle' columns (no more reuse of
    the legacy '*_shoulder_angle' name).
"""

import numpy as np
import mujoco

from simulation.config import pipeline as P
from simulation.config.parameters import parameters
from simulation.core.data_loader import load_model


def angles_to_trajectory(df, frequency, xml_path=None, parameters_=None):
    """
    Build qpos / qvel arrays from a joint-angle DataFrame.

    Args:
        df (pd.DataFrame): must contain 'T13_Y', 'T13_Z' (mm) and the 12
                           '<leg>_<joint>_angle' columns (degrees).
        frequency (float): capture rate (Hz); sets the finite-difference dt.
        xml_path         : model XML (default config.XML_PATH).
        parameters_      : geometry params (default config.parameters).

    Returns:
        dict with 'qpos' (T,19), 'qvel' (T,18), 'time' (T,), 'frequency'.
    """
    xml_path = xml_path or P.XML_PATH
    parameters_ = parameters_ if parameters_ is not None else parameters

    model, data = load_model(xml_path, parameters_)
    mujoco.mj_resetData(model, data)
    torso_quat = data.qpos[3:7].copy()  # identity orientation from reset

    # Root (torso) translation from the T13 marker: mm -> m, with Z re-zeroed.
    root_y = (df[P.ROOT_MARKER + "_Y"].to_numpy(np.float64)) * P.MM_TO_M
    root_z = (df[P.ROOT_MARKER + "_Z"].to_numpy(np.float64) - P.T13_Z_OFFSET_MM) * P.MM_TO_M

    n = len(df)
    qpos = np.zeros((n, model.nq))

    for i in range(n):
        # Mocap moves forward along Y; map to robot +X. Lateral (robot Y) = 0.
        data.qpos[0:3] = [root_y[i], 0.0, root_z[i]]
        data.qpos[3:7] = torso_quat
        data.qvel[:] = 0.0

        for leg in P.LEG_ORDER:
            for joint in P.JOINT_ORDER:
                ang = np.radians(df[f"{leg}_{joint}_angle"].iloc[i])
                data.joint(f"{leg}_{joint}_joint").qpos = ang

        mujoco.mj_forward(model, data)
        qpos[i] = data.qpos.copy()

    qvel = _finite_difference_qvel(qpos, frequency)
    time = np.arange(n) / frequency

    return {"qpos": qpos, "qvel": qvel, "time": time, "frequency": float(frequency)}


def _finite_difference_qvel(qpos, frequency):
    """
    Generalized velocity from qpos by finite difference.

    qpos layout: [pos(3), quat(4), joints(12)]  -> qvel: [vlin(3), wang(3), vjoint(12)].
    Angular velocity uses the small-angle quaternion approximation w ~= 2*d/dt[qx,qy,qz].
    """
    dt = 1.0 / frequency
    dq = np.gradient(qpos, axis=0) / dt
    vel_linear  = dq[:, 0:3]
    vel_angular = 2.0 * dq[:, 4:7]   # qx,qy,qz -> cols 4,5,6
    vel_joints  = dq[:, 7:]
    return np.concatenate([vel_linear, vel_angular, vel_joints], axis=1)


def save_trajectory(traj, path=None):
    """Save a trajectory dict to .npz (default config PROC_DIR / TRAJ_NAME)."""
    path = path or (P.PROC_DIR / P.TRAJ_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        time=traj["time"],
        qpos=traj["qpos"],
        qvel=traj["qvel"],
        frequency=traj["frequency"],
    )
    return path
