# simulation/movement/walking.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import mujoco
from simulation.config import parameters
from simulation.core.data_loader import load_model, interpolate
from simulation.core.utils import image_renderer, video_renderer

# ==============================================================================
# PATH CONFIGURATION (single source of truth: simulation/config/paths.py)
# ==============================================================================
from simulation.config import paths

XML_PATH = paths.XML_DIR / 'senecabot_move.xml'
CSV_PATH = paths.PROC_DIR / 'ovino_angles.csv'
OUTPUT_DIR = paths.FIGURES

# List of columns to extract and process from the CSV file
ANGLE_COLS = [
    "fl_hip_angle",  "fl_knee_angle",  "fl_ankle_angle",
    "fr_hip_angle",  "fr_knee_angle",  "fr_ankle_angle",
    "bl_hip_angle",  "bl_knee_angle",  "bl_ankle_angle",
    "br_hip_angle",  "br_knee_angle",  "br_ankle_angle",
]

def walk(framerate=60):
    """
    Executes a walking simulation in MuJoCo based on interpolated motion data.

    Args:
        framerate (int): The target framerate for the output video. Defaults to 60.

    Returns:
        tuple: A tuple containing two lists (positions, velocities) recorded 
               at each simulation step.
    """
    # --- 1. Load Model and Initialize Renderer ---
    model, data = load_model(XML_PATH, parameters.parameters)
    renderer = mujoco.Renderer(model, height=720, width=1280)

    # --- 2. Load and Preprocess CSV Data ---
    df_raw = pd.read_csv(CSV_PATH)
    
    # Scale coordinates to meters (assuming original data is in mm)
    df_raw["T13_Y"] = df_raw["T13_Y"] / 1000
    df_raw["T13_Z"] = (df_raw["T13_Z"] - 350) / 1000

    # --- 3. Interpolate Data to Match MuJoCo Timestep ---
    model.opt.timestep = 0.0001
    target_hz = 1.0 / model.opt.timestep
    df = interpolate(df_raw, ANGLE_COLS, target_hz=target_hz)

    # Re-interpolate trajectory values to match the new time scale
    t_original = df_raw["Tiempo_s"].to_numpy(dtype=np.float64)
    t_new = df["Tiempo_s"].to_numpy(dtype=np.float64)
    from scipy.interpolate import interp1d
    df["T13_Y"] = interp1d(t_original, df_raw["T13_Y"].to_numpy(), kind="cubic")(t_new)
    df["T13_Z"] = interp1d(t_original, df_raw["T13_Z"].to_numpy(), kind="cubic")(t_new)

    # Lists to store simulation states and render frames
    frames = []
    positions = []
    velocities = []

    mujoco.mj_resetData(model, data)

    # --- 4. Set Initial Positioning ---
    # Apply spatial translations
    data.joint("root_x").qpos[0] = df["T13_Y"].iloc[0]
    data.joint("root_z").qpos[0] = df["T13_Z"].iloc[0] - 0.5

    # Apply initial joint angles
    for col in ANGLE_COLS:
        # Map CSV column names to corresponding MuJoCo joint names
        joint_name = col.replace("_angle", "")
        data.joint(joint_name).qpos[0] = np.radians(df[col].iloc[0])

    mujoco.mj_forward(model, data)

    # --- 5. Main Simulation Loop ---
    for i in range(len(df)):
        # Apply interpolated control signals (angles in radians) to actuators
        data.actuator("fl_hip").ctrl[0]   = np.radians(df["fl_hip_angle"].iloc[i])
        data.actuator("fl_knee").ctrl[0]  = np.radians(df["fl_knee_angle"].iloc[i])
        data.actuator("fl_ankle").ctrl[0] = np.radians(df["fl_ankle_angle"].iloc[i])

        data.actuator("fr_hip").ctrl[0]   = np.radians(df["fr_hip_angle"].iloc[i])
        data.actuator("fr_knee").ctrl[0]  = np.radians(df["fr_knee_angle"].iloc[i])
        data.actuator("fr_ankle").ctrl[0] = np.radians(df["fr_ankle_angle"].iloc[i])

        data.actuator("bl_hip").ctrl[0]   = np.radians(df["bl_hip_angle"].iloc[i])
        data.actuator("bl_knee").ctrl[0]  = np.radians(df["bl_knee_angle"].iloc[i])
        data.actuator("bl_ankle").ctrl[0] = np.radians(df["bl_ankle_angle"].iloc[i])

        data.actuator("br_hip").ctrl[0]   = np.radians(df["br_hip_angle"].iloc[i])
        data.actuator("br_knee").ctrl[0]  = np.radians(df["br_knee_angle"].iloc[i])
        data.actuator("br_ankle").ctrl[0] = np.radians(df["br_ankle_angle"].iloc[i])

        # Step the physics engine forward
        mujoco.mj_step(model, data)

        # Record kinematic state
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())

        # Render frame if enough time has passed to meet target framerate
        if data.time * framerate >= len(frames):
            renderer.update_scene(data, camera="follow")
            frames.append(renderer.render().copy())

    renderer.close()

    # --- 6. Export Results ---
    video_filename = OUTPUT_DIR / "walking_simulation.mp4"
    
    # Render and save the collected frames to the output directory
    video_renderer(frames, fps=framerate, output_path=str(video_filename))

    print(f"Video saved successfully to: {video_filename}")
    return positions, velocities