"""
trajectory_generation.py

This script processes and adapts a single reference trajectory for Loco-Mujoco.
It loads the raw mocap trajectory (trajectory.npz, or the opt-in
trajectory_augmented.npz), runs MuJoCo forward kinematics on a standalone model
loaded from the robot XML to compute the site/body quantities Loco-MuJoCo needs,
and saves the result as trajectory_adapted.npz.

It uses Hydra for configuration management and MuJoCo for kinematic computations.
No Loco-MuJoCo environment is built here, so generation no longer depends on a
previously generated trajectory_adapted.npz.
"""

import jax.numpy as jnp
import numpy as np
import mujoco
from pathlib import Path

# Loco-Mujoco imports
from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData

import hydra
from omegaconf import DictConfig, OmegaConf

# Single source of truth for paths, layout and capture frequency.
from simulation.config import pipeline as P
from simulation.config import paths


def unpack_loco_trajectory(file_path: Path) -> dict:
    """
    Loads and unpacks trajectory data from a .npz file.

    Supports both the original trajectory.npz (single clip) and
    trajectory_augmented.npz (multiple clips with split_points).

    The capture frequency is read from the file (the 'frequency' field written by
    the mocap pipeline), falling back to deriving it from a single 'time' segment,
    then to config.POINT_RATE. The old hardcoded 2000 Hz was the analog/force-plate
    rate, not the kinematic sample rate.
    """
    with np.load(file_path, allow_pickle=True) as loaded_data:
        qpos = loaded_data['qpos']
        qvel = loaded_data['qvel']
        split_points = (loaded_data['split_points'].tolist()
                        if 'split_points' in loaded_data
                        else [0, qpos.shape[0]])

        if 'frequency' in loaded_data:
            frequency = float(loaded_data['frequency'])
        elif 'time' in loaded_data and len(split_points) <= 2:
            # Single contiguous clip: derive from the time axis.
            t = np.asarray(loaded_data['time'], dtype=np.float64)
            frequency = float(1.0 / np.mean(np.diff(t)))
        else:
            frequency = float(P.POINT_RATE)

        print("--- Unpacking Summary ---")
        print(f"File: {file_path}")
        print(f"Number of steps (Timesteps): {qpos.shape[0]}")
        print(f"qpos dimensions: {qpos.shape}")
        print(f"qvel dimensions: {qvel.shape}")
        print(f"Segments (split_points): {len(split_points) - 1}")
        print(f"Capture frequency: {frequency} Hz")
        print("-" * 34)

        return {
            "qpos": qpos,
            "qvel": qvel,
            "freq": frequency,
            "split_points": split_points,
        }

# ==============================================================================
# HYDRA WRAPPER AND MAIN FLOW
# ==============================================================================
@hydra.main(version_base=None, config_path="../../training", config_name="conf")
def main(config: DictConfig):
    """
    Main execution function managed by Hydra. Initializes the environment,
    processes the trajectory kinematics, and records a video playback.
    """
    # Convert the Hydra configuration to a native dictionary
    config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    
    # --------------------------------------------------------------------------
    # PATH CONFIGURATION (single source of truth: simulation/config/paths.py)
    # --------------------------------------------------------------------------
    OUTPUT_DIR = paths.FIGURES
    
    # --------------------------------------------------------------------------
    # STANDALONE MUJOCO MODEL (no Loco-MuJoCo env)
    # --------------------------------------------------------------------------
    # The kinematic processing below only needs a MuJoCo `model`/`data` for forward
    # kinematics, so we load the model directly from the XML. We deliberately do NOT
    # build a Loco-MuJoCo env here: ImitationFactory requires a seed trajectory that
    # already contains site data, which used to force this script to load its own
    # previous output (trajectory_adapted.npz) as the seed -- a circular dependency
    # that (a) broke a clean checkout and (b) let a stale/concatenated trajectory leak
    # into the env. Standalone FK reproduces the env's site_xpos/xpos to float32
    # precision, so exactly one trajectory -- the raw input below -- flows through.
    model = mujoco.MjModel.from_xml_path(str(P.XML_PATH))
    data = mujoco.MjData(model)

    # Load the raw trajectory to be processed.
    # Uses the base trajectory.npz by default. The augmented dataset is strictly
    # opt-in (it is NOT auto-detected), so a stale trajectory_augmented.npz can
    # never sneak into training:
    #     python trajectory_generation.py +use_augmented=true
    _proc = paths.PROC_DIR
    use_augmented = bool(OmegaConf.select(config, "use_augmented", default=False))
    TRAJ_PATH = _proc / (P.AUGMENTED_NAME if use_augmented else P.TRAJ_NAME)
    if use_augmented and not TRAJ_PATH.exists():
        raise FileNotFoundError(
            f"use_augmented=true but {TRAJ_PATH.name} not found. "
            f"Build it first with: python -m simulation.mocap.run --augment"
        )
    print(f"Using trajectory file: {TRAJ_PATH.name}")
    datos = unpack_loco_trajectory(TRAJ_PATH)
    
    # Joint nomenclature in qpos. The "_joint" suffixes are included to 
    # maintain consistency with the model's XML schema.
    joint_names_qpos = [
        "root",         # Indices 0-6 corresponding to position (3) and quaternion (4)
        "fr_hip_joint", "fr_knee_joint", "fr_ankle_joint", 
        "fl_hip_joint", "fl_knee_joint", "fl_ankle_joint",
        "br_hip_joint", "br_knee_joint", "br_ankle_joint",
        "bl_hip_joint", "bl_knee_joint", "bl_ankle_joint"
    ]
    
    qpos        = datos["qpos"]
    qvel        = datos["qvel"]
    split_points = datos["split_points"]
    capture_freq = datos["freq"]

    # --------------------------------------------------------------------------
    # SEAMLESS LOOPING REFERENCE (default on)
    # --------------------------------------------------------------------------
    # The env never terminates at the clip end; it WRAPS the single sub-trajectory,
    # and the goal observation goes NaN at the wrap (intrinsic + policy-independent
    # -- see CLAUDE.md). A lone ~2 s clip therefore truncates every episode at the
    # wrap and prevents sustained-gait learning. Fix: tile a clean steady-state gait
    # cycle into ONE long contiguous clip -- internal cycle-joins are normal frames,
    # only the single final boundary wraps -- so episodes run full-length while
    # random-start (RSI) is preserved. Disable with: +make_cyclic=false
    make_cyclic = bool(OmegaConf.select(config, "make_cyclic", default=True))
    loop_duration_s = float(OmegaConf.select(config, "loop_duration_s", default=60.0))
    # Enforce left/right (bilateral) symmetry on the gait cycle (default on). The
    # raw ovine reference has a per-joint postural bias between the left and right
    # legs (front-left ~5-9 deg more flexed) that reads visually as an fl_knee/
    # fl_foot "trip"; symmetrization removes it. Disable with: +symmetrize_gait=false
    # Mode: average (default) | mirror_right | mirror_left.
    symmetrize_gait = bool(OmegaConf.select(config, "symmetrize_gait", default=True))
    symmetrize_mode = str(OmegaConf.select(config, "symmetrize_mode", default="average"))
    # Root forward speed (2026-09-27). The mocap root travel is the SHEEP's T13 speed
    # (~1.42 m/s), ~7x what the robot's shorter legs can push without foot slip; agents
    # trained on it learned micro-hops. Default "no_slip" rescales the root XY travel to the
    # speed the leg motion supports (see cyclic.no_slip_root_speed). Restore the old
    # reference with +root_speed=mocap, or set a speed in m/s with +root_speed=0.3.
    root_speed = OmegaConf.select(config, "root_speed", default="no_slip")
    if make_cyclic and not use_augmented:
        from simulation.mocap.cyclic import make_looping_reference
        loop = make_looping_reference(np.asarray(qpos), np.asarray(qvel),
                                      capture_freq, loop_duration_s=loop_duration_s,
                                      symmetrize=symmetrize_gait,
                                      symmetrize_mode=symmetrize_mode,
                                      root_speed=root_speed, model=model)
        qpos, qvel, split_points = loop["qpos"], loop["qvel"], loop["split_points"]
        _i = loop["info"]
        print(f"--- Looping reference: cycle frames [{_i['start']}, {_i['start']+_i['period']}] "
              f"(period {_i['period']}, seam {_i['seam_deg']:.1f} deg), tiled x{_i['n_tiles']} "
              f"-> {_i['frames']} frames (~{_i['frames']/capture_freq:.1f}s @ {capture_freq:.0f}Hz)")
        print(f"    Root speed: {root_speed} -> {_i['root_speed']:.3f} m/s "
              f"(mocap {_i['root_speed_mocap']:.3f} m/s, x{_i['root_speed_scale']:.3f})")
        if "no_slip" in _i:
            print(f"    no-slip: stance foot vx rel. body (FR,FL,BR,BL) = "
                  f"{np.round(_i['no_slip']['stance_vx'], 3).tolist()} m/s, "
                  f"x sweep = {np.round(_i['no_slip']['sweep_x'], 3).tolist()} m")
        if _i.get("symmetrized"):
            print(f"    L/R symmetrization: mode={_i['symmetrize_mode']}, "
                  f"removed peak left/right joint bias of {_i['lr_bias_deg_prefix']:.1f} deg")
        else:
            print("    L/R symmetrization: OFF")
    elif make_cyclic and use_augmented:
        print("make_cyclic skipped: incompatible with use_augmented (multi-clip dataset).")

    print("qpos shape:", qpos.shape)
    print("qvel shape:", qvel.shape)
    print("model.nq:", model.nq)
    print("model.nv:", model.nv)
    
    # --------------------------------------------------------------------------
    # KINEMATIC PROCESSING OF THE TRAJECTORY
    # --------------------------------------------------------------------------
    # Iterative calculation of positions (site_xpos) and rotation matrices 
    # (site_xmat) for each timestep using MuJoCo.
    # Read the mimic site names directly from the compiled model (the XML) rather
    # than hardcoding them: every reward/tracking site is named "*_mimic" (class
    # "mimic" in senecabot_loco.xml), so we enumerate the model's sites and keep
    # those, in model declaration order. Adding/removing a "*_mimic" site in the
    # XML now propagates here automatically.
    site_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        for i in range(model.nsite)
    ]
    site_names = [s for s in site_names if s and s.endswith("_mimic")]
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in site_names]
    
    n_steps = qpos.shape[0]
    n_sites = len(site_ids)
    n_bodies = model.nbody
    
    all_site_xpos   = np.zeros((n_steps, n_sites, 3))
    all_site_xmat   = np.zeros((n_steps, n_sites, 9))
    all_xpos        = np.zeros((n_steps, n_bodies, 3))
    all_xquat       = np.zeros((n_steps, n_bodies, 4))
    all_cvel        = np.zeros((n_steps, n_bodies, 6))
    all_subtree_com = np.zeros((n_steps, n_bodies, 3))
    
    for i in range(n_steps):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        
        # Computation of forward kinematics and center of mass in MuJoCo
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        mujoco.mj_crb(model, data)
        
        for j, sid in enumerate(site_ids):
            all_site_xpos[i, j] = data.site_xpos[sid]
            all_site_xmat[i, j] = data.site_xmat[sid].flatten()
            
        all_xpos[i]        = data.xpos
        all_xquat[i]       = data.xquat
        all_cvel[i]        = data.cvel
        all_subtree_com[i] = data.subtree_com
        
    # Construction of the data structures required by Loco-Mujoco
    njnt = model.njnt
    jnt_type = model.jnt_type
    
    # Store the TRUTHFUL capture frequency, not the control rate. The data here is
    # still at the mocap/augment rate; labelling it as the control rate (the old
    # `frequency=1/env.dt`) told Loco-MuJoCo "already at control_dt" and skipped
    # interpolation, scaling every velocity wrong (the qvel/site_rvel DTW blowups
    # in TODO_time_consistency_check.md). With the true frequency, the trajectory
    # handler interpolates from traj_dt -> control_dt correctly.
    #
    # The control rate is read from config (P.CONTROL_RATE) rather than 1/env.dt:
    # this script no longer builds an env, and the value is purely informational
    # (the stored frequency is capture_freq; Loco-MuJoCo does the real interpolation
    # to whatever control_dt the env uses at load time).
    control_freq = P.CONTROL_RATE
    print("--- Frequency consistency ---")
    print(f"  Trajectory capture frequency : {capture_freq} Hz  (stored)")
    print(f"  Env control frequency (config): {control_freq:.3f} Hz")
    if abs(capture_freq - control_freq) > 1e-6:
        print(f"  -> Loco-MuJoCo will interpolate {capture_freq} Hz -> {control_freq:.3f} Hz")
    print("-" * 34)

    traj_info = TrajectoryInfo(
        joint_names_qpos,
        model=TrajectoryModel(
            njnt,
            jnp.array(jnt_type),
            nsite=model.nsite,
            site_bodyid=jnp.zeros((0,), dtype=jnp.int32),
            site_pos=jnp.zeros((0, 3)),
            site_quat=jnp.zeros((0, 4)),
        ),
        frequency=capture_freq,
        site_names=site_names
    )
    
    traj_data = TrajectoryData(
        jnp.array(qpos),
        jnp.array(qvel),
        xpos=jnp.array(all_xpos),
        xquat=jnp.array(all_xquat),
        cvel=jnp.array(all_cvel),
        subtree_com=jnp.array(all_subtree_com),
        site_xpos=jnp.array(all_site_xpos),
        site_xmat=jnp.array(all_site_xmat),
        split_points=jnp.array(split_points)
    )
    
    # Save the processed trajectory
    SAVE_PATH = paths.PROC_DIR / 'trajectory_adapted'
    traj = Trajectory(traj_info, traj_data)
    traj.save(SAVE_PATH)
    print(f"Trajectory successfully saved at: {SAVE_PATH}.npz")

if __name__ == "__main__":
    main()