"""
High-quality MuJoCo renders of the SenecaBot training model
(simulation/assets/xml/senecabot_loco.xml) for the thesis.

Produces three views — front, side, diagonal — as high-resolution PNGs in
final_document/images/ following the repository naming convention
(<category>__<descriptor>__<variant>.png) and logging to images/_manifest.tsv.

These are 3D renders (not CSV plots), so they live outside the generate_all.py
matplotlib pipeline but reuse io_utils for naming + manifest bookkeeping.

Run from anywhere (paths are resolved relative to this file):
    MUJOCO_GL=glfw python final_document/scripts/render_model.py
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import time
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image

import io_utils  # noqa: E402  (sibling module; run from scripts/ dir)

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
XML = (SCRIPTS_DIR.parent.parent / "simulation" / "assets" / "xml"
       / "senecabot_loco.xml")

WIDTH, HEIGHT = 1920, 1440          # bounded by model offwidth/offheight (3840x2160)
CATEGORY, DESCRIPTOR = "schematic", "robot_model"

# Canonical standing pose (degrees, joint order hip/knee/ankle), taken verbatim
# from the project's reference visualization
# (notebooks/exploration/Body_display.ipynb "Standing pose"). Front/back legs use
# the model's opposite sign conventions; this yields a natural articulated stance.
FRONT_STANCE = (82, 95, 13)
REAR_STANCE = (-78, -120, -42)
JOINTS = ("hip", "knee", "ankle")

# Free-camera setups: (azimuth, elevation, distance, lookat_z)
VIEWS = {
    "front":    dict(azimuth=180, elevation=-12, distance=1.55, lookz=0.18),
    "side":     dict(azimuth=90,  elevation=-12, distance=1.70, lookz=0.18),
    "diagonal": dict(azimuth=-130, elevation=-20, distance=1.75, lookz=0.18),
}


def build_data(model: mujoco.MjModel) -> mujoco.MjData:
    """Pose the robot in the canonical standing stance, resting on the floor."""
    data = mujoco.MjData(model)
    for leg, stance in (("fr", FRONT_STANCE), ("fl", FRONT_STANCE),
                        ("br", REAR_STANCE), ("bl", REAR_STANCE)):
        for joint, deg in zip(JOINTS, stance):
            data.joint(f"{leg}_{joint}_joint").qpos = np.deg2rad(deg)
    data.qpos[0:3] = [0.0, 0.0, 0.35]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    # Drop the base so the lowest foot sphere just touches the floor (z=0).
    foot_geoms = [f"{leg}_foot" for leg in ("fr", "fl", "br", "bl")]
    zmin = min(
        data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)][2]
        - model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)][0]
        for g in foot_geoms
    )
    data.qpos[2] -= zmin
    mujoco.mj_forward(model, data)
    return data


def make_options() -> mujoco.MjvOption:
    """Hide all sites (mimic markers + foot sites) for a clean model figure."""
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0            # disable every site group
    return opt


def render_view(model, data, opt, name, cfg) -> Path:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, cfg["lookz"]]
    cam.azimuth = cfg["azimuth"]
    cam.elevation = cfg["elevation"]
    cam.distance = cfg["distance"]

    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        renderer.update_scene(data, camera=cam, scene_option=opt)
        pixels = renderer.render()

    fig_id = io_utils.build_fig_id(CATEGORY, DESCRIPTOR, name)
    out = io_utils.IMG_DIR / f"{fig_id}.png"
    io_utils.IMG_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(out)

    # Manifest bookkeeping (mirrors io_utils.save_figure for non-matplotlib output).
    io_utils._append_manifest(
        fig_id=fig_id, out=out, sources=[str(XML.name)],
        plotter="render_model.render_view",
        kwargs={"view": name, **cfg, "size": [WIDTH, HEIGHT]},
    )
    return out


def main() -> None:
    print(f"Loading {XML} (MUJOCO_GL={os.environ['MUJOCO_GL']})")
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = build_data(model)
    opt = make_options()
    for name, cfg in VIEWS.items():
        out = render_view(model, data, opt, name, cfg)
        print(f"  wrote {out.relative_to(SCRIPTS_DIR.parent)}  ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
