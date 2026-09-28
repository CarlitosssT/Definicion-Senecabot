"""
io.py

Read raw C3D motion-capture files into a tidy marker DataFrame, and report the
*true* capture frequency straight from the file header (never assumed).
"""

import os
import numpy as np
import pandas as pd

import ezc3d

from simulation.config import pipeline as P


def list_c3d_files(c3d_dir=None):
    """Return the sorted list of .c3d filenames in ``c3d_dir`` (default config)."""
    c3d_dir = c3d_dir or P.C3D_DIR
    return sorted(f for f in os.listdir(c3d_dir) if f.endswith(".c3d"))


def load_c3d(path, markers=None):
    """
    Load a C3D file into a DataFrame of selected marker coordinates.

    Args:
        path (str | Path): path to the .c3d file.
        markers (list)   : marker labels to extract. Defaults to config.MARKERS.

    Returns:
        (df, point_rate):
            df         : DataFrame with 'Tiempo_s' and '<marker>_{X,Y,Z}' columns.
            point_rate : true marker sample rate (Hz) read from the C3D header.

    Raises:
        KeyError: if any requested marker is missing from the file.
    """
    markers = markers or P.MARKERS
    c3d = ezc3d.c3d(str(path))

    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    points = c3d["data"]["points"]              # (4, n_markers, n_frames)
    point_rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
    n_frames = points.shape[2]

    time = np.arange(n_frames) / point_rate
    out = {"Tiempo_s": time}

    missing = [m for m in markers if m not in labels]
    if missing:
        raise KeyError(f"Markers not found in {path}: {missing}")

    for m in markers:
        idx = labels.index(m)
        out[f"{m}_X"] = points[0, idx, :]
        out[f"{m}_Y"] = points[1, idx, :]
        out[f"{m}_Z"] = points[2, idx, :]

    return pd.DataFrame(out), point_rate


def describe_c3d(path):
    """Print a short header summary (markers, frames, rate, duration)."""
    c3d = ezc3d.c3d(str(path))
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    n_frames = c3d["data"]["points"].shape[2]
    rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
    print(f"File     : {os.path.basename(str(path))}")
    print(f"Markers  : {len(labels)}")
    print(f"Frames   : {n_frames}")
    print(f"Rate     : {rate} Hz")
    print(f"Duration : {n_frames / rate:.3f} s")
    return rate
