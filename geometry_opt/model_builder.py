"""
Build the MuJoCo model of a candidate design.

A design is the thesis model plus:
  * one actuator point mass per joint, placed exactly at the joint (origin of the body the joint
    drives). A point mass ON the joint axis contributes no rotational inertia about that joint, so
    attaching it to the child or the parent body gives the same dynamics.
  * (from iteration 1) the leg links as hollow tubes of the chosen material: the thesis' solid
    capsules keep their geometry (collisions) but get zero mass, and a massless-collision cylinder
    of the tube's outer radius carries the tube mass. MuJoCo computes that cylinder's inertia as a
    solid cylinder of the same mass; for these slender links (L >> R) the transverse inertia is
    dominated by m L^2 / 12 so the error is negligible.
  * the foot contact sphere printed in the same material (FOOT_SPHERE_INFILL).
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import config as C

LEGS = ("fr", "fl", "br", "bl")
SEGS = ("upper", "lower", "foot")
JOINT_OF_SEG = {"upper": "hip", "lower": "knee", "foot": "ankle"}
JOINT_NAMES = tuple(f"{l}_{JOINT_OF_SEG[s]}" for l in LEGS for s in SEGS)   # actuator order
LINK_NAMES = tuple(f"{l}_{s}" for l in LEGS for s in SEGS)                   # body driven by that joint


@dataclass
class Design:
    material: str | None = None                          # None = thesis model (capsules, water density)
    motor_mass_kg: dict = field(default_factory=dict)    # joint name (e.g. "fr_knee") -> kg
    tube_ro_m: dict = field(default_factory=dict)        # link name (e.g. "fr_lower") -> outer radius

    def key(self):
        return (self.material, tuple(sorted(self.motor_mass_kg.items())),
                tuple(sorted((k, round(v, 6)) for k, v in self.tube_ro_m.items())))


def tube_area(ro, t):
    ri = max(ro - t, 0.0)
    return np.pi * (ro ** 2 - ri ** 2)


def _capsule(body):
    caps = [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_CAPSULE]
    assert len(caps) == 1, body.name
    return caps[0]


def segment_lengths(xml=C.XML_PATH):
    spec = mujoco.MjSpec.from_file(str(xml))
    out = {}
    for link in LINK_NAMES:
        ft = np.asarray(_capsule(spec.body(link)).fromto)
        out[link] = float(np.linalg.norm(ft[3:] - ft[:3]))
    return out


def build_spec(design: Design, xml=C.XML_PATH):
    spec = mujoco.MjSpec.from_file(str(xml))
    t = C.MIN_WALL_THICKNESS_MM / 1000.0
    for link in LINK_NAMES:
        body = spec.body(link)
        if design.material is not None and C.STRUCTURE_MASS_IN_LOOP and link in design.tube_ro_m:
            rho = C.MATERIALS[design.material]["density_kg_m3"]
            cap = _capsule(body)
            ft = np.asarray(cap.fromto, dtype=float)
            L = float(np.linalg.norm(ft[3:] - ft[:3]))
            ro = float(design.tube_ro_m[link])
            cap.mass = 0.0                                              # keep geometry for contacts
            body.add_geom(name=f"{link}_tube", type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=ft,
                          size=[ro, 0, 0], mass=rho * tube_area(ro, t) * L,
                          contype=0, conaffinity=0, group=3)
            if link.endswith("_foot"):                                   # contact sphere
                sph = [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_SPHERE and g.name == link][0]
                r = float(sph.size[0])
                sph.mass = rho * 4.0 / 3.0 * np.pi * r ** 3 * C.FOOT_SPHERE_INFILL
        joint = f"{link[:2]}_{JOINT_OF_SEG[link[3:]]}"
        if design.motor_mass_kg.get(joint, 0.0) > 0.0:
            body.add_geom(name=f"{joint}_motor", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0],
                          pos=[0, 0, 0], mass=float(design.motor_mass_kg[joint]),
                          contype=0, conaffinity=0, group=3)
    if C.JOINT_DAMPING_OVERRIDE is not None:
        for jn in JOINT_NAMES:
            spec.joint(f"{jn}_joint").damping = [float(C.JOINT_DAMPING_OVERRIDE), 0.0, 0.0]  # [linear, ...] in MuJoCo >= 3.8
    return spec


def build_model(design: Design, xml=C.XML_PATH):
    return build_spec(design, xml).compile()


def mass_breakdown(model, design: Design):
    """Total / torso / motors / leg-structure masses [kg] of a compiled design."""
    total = float(model.body_subtreemass[model.body("base").id])
    motors = float(sum(design.motor_mass_kg.values()))
    torso = float(model.body_mass[model.body("base").id])
    return dict(total=total, torso=torso, motors=motors, structure=total - torso - motors)
