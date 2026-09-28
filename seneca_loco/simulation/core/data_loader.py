import os
import mujoco
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

def load_model(xml_path, parameters):
    """
    Loads a templated MuJoCo model, filling in the geometric ``parameters``.

    The XML is compiled with the working directory temporarily switched to the
    file's own folder, so relative includes (``scene.xml``, ``mimic_sites.xml``)
    and mesh assets resolve correctly regardless of the caller's cwd.

    Args:
        xml_path (str | Path): path to the templated XML file.
        parameters (dict)    : substitution values for the XML ``{placeholders}``.

    Returns:
        (mujoco.MjModel, mujoco.MjData)
    """
    xml_path = Path(xml_path)
    with open(xml_path, "r") as file:
        xml_temp = file.read()
    xml = xml_temp.format(**parameters)

    original_cwd = os.getcwd()
    try:
        os.chdir(xml_path.parent)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
    finally:
        os.chdir(original_cwd)
    return model, data

def interpolate(df, angle_cols, time_col="Tiempo_s", target_hz=500, kind="cubic"):
    """
    Interpolates joint angle columns in a DataFrame to a target frequency.

    Args:
        df (pd.DataFrame)    : Original DataFrame with motion capture data.
        angle_cols (list)    : List of column names to interpolate.
        time_col (str)       : Name of the time column. Default "Tiempo_s".
        target_hz (float)    : Target frequency in Hz. Default 500 (MuJoCo 0.002s timestep).
        kind (str)           : Interpolation kind passed to scipy interp1d. Default "cubic".

    Returns:
        pd.DataFrame: New DataFrame interpolated at target_hz.
    """
    t_original = df[time_col].to_numpy(dtype=np.float64)
    t_new = np.arange(t_original[0], t_original[-1], 1.0 / target_hz)

    df_interp = {time_col: t_new}
    for col in angle_cols:
        f = interp1d(t_original, df[col].to_numpy(dtype=np.float64), kind=kind)
        df_interp[col] = f(t_new)

    return pd.DataFrame(df_interp)


def resample_to_n(arr, n_target, kind="cubic"):
    """
    Resample a (T, D) array along axis 0 to ``n_target`` rows, column by column.

    Shared resampling primitive used by the augmentation stage (cycle
    normalization and time scaling) so there is one interpolation code path.

    Args:
        arr (np.ndarray): shape (T, D).
        n_target (int)  : number of output rows.
        kind (str)      : scipy interp1d kind. Default "cubic".

    Returns:
        np.ndarray of shape (n_target, D).
    """
    arr = np.asarray(arr, dtype=np.float64)
    old_t = np.linspace(0.0, 1.0, arr.shape[0])
    new_t = np.linspace(0.0, 1.0, n_target)
    return np.stack(
        [interp1d(old_t, arr[:, j], kind=kind)(new_t) for j in range(arr.shape[1])],
        axis=1,
    )