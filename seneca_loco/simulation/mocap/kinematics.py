"""
kinematics.py

Turn a marker DataFrame (from io.load_c3d) into joint angles for all four legs,
using the declarative marker/sign map in ``config.pipeline.LEG_CHAINS``.

Angles are computed in the sagittal (Y-Z) plane. Replaces the hand-written,
per-leg angle block that used to live in Ovine_data_visualizer.ipynb.
"""

import numpy as np

from simulation.config import pipeline as P


def calc_joint_angle(df, marker_a, marker_b, sign_z, sign_y):
    """
    Absolute angle (degrees) of the segment marker_a -> marker_b in the Y-Z plane.

        angle = atan2(sign_y * (B.y - A.y), sign_z * (B.z - A.z))
    """
    dz = sign_z * (df[f"{marker_b}_Z"] - df[f"{marker_a}_Z"])
    dy = sign_y * (df[f"{marker_b}_Y"] - df[f"{marker_a}_Y"])
    return np.degrees(np.arctan2(dy, dz))


def _unwrap_deg(series):
    """Remove 360-degree jumps from an angle series (degrees in, degrees out)."""
    return np.degrees(np.unwrap(np.radians(series)))


def _wrap_180(series):
    """Wrap angles into [-180, 180)."""
    return (series + 180.0) % 360.0 - 180.0


def compute_joint_angles(df):
    """
    Add the 12 joint-angle columns to ``df`` and return it.

    Columns follow config.angle_columns(), e.g. 'fl_hip_angle', 'fl_knee_angle',
    'fl_ankle_angle', ... The middle segment is named 'knee' (it drives the
    robot knee joint), not the legacy 'shoulder'.
    """
    for leg, chain in P.LEG_CHAINS.items():
        # hip: absolute angle of the proximal segment
        hip = _unwrap_deg(calc_joint_angle(df, *chain["hip"]))

        # knee: relative to hip
        knee = _unwrap_deg(calc_joint_angle(df, *chain["knee"]) - hip)

        # ankle: relative chain, per-leg combination of hip & knee
        a_marker_a, a_marker_b, a_sz, a_sy, hip_c, knee_c = chain["ankle"]
        ankle_raw = calc_joint_angle(df, a_marker_a, a_marker_b, a_sz, a_sy) \
            + hip_c * hip + knee_c * knee
        ankle = _unwrap_deg(ankle_raw)

        df[f"{leg}_hip_angle"]   = hip
        df[f"{leg}_knee_angle"]  = knee
        df[f"{leg}_ankle_angle"] = ankle

    # Wrap all joint angles into [-180, 180)
    for col in P.angle_columns():
        df[col] = _wrap_180(df[col])

    return df


def angle_summary(df):
    """Return a min/max table (degrees) for every joint angle column."""
    cols = P.angle_columns()
    return (df[cols].agg(["min", "max"]).T
            .rename(columns={"min": "Min (deg)", "max": "Max (deg)"})
            .round(2))
