"""
mocap pipeline subpackage.

Turns raw ovine motion capture into a Loco-MuJoCo trajectory, in stages:

    io          C3D file        -> marker DataFrame + true capture rate
    kinematics  markers         -> joint-angle DataFrame
    build       joint angles    -> qpos / qvel trajectory.npz (via MuJoCo FK)
    augment     trajectory.npz  -> trajectory_augmented.npz (phase/mirror/time/noise)
    run         CLI tying the above together with a time-consistency report

All paths, the capture frequency, and the marker/sign conventions come from
``simulation.config.pipeline`` so there is a single source of truth.
"""
