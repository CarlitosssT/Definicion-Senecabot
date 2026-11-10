"""
trajectory_generation_rr.py

Adapts the reference gait (trajectory_adapted.npz, RRR model: hip + knee + ankle) to the RR model with the two RR
bar mechanisms (senecabot_loco_rr.xml) and saves trajectory_adapted_rr.npz.

    python -m simulation.mocap.trajectory.trajectory_generation_rr [--policy v140]

Only knee and ankle are replicated. The hip no longer exists, so the knee of the RR model is the absolute pitch of the
tibia with respect to the torso in the reference (knee_RR = -pitch_tibia, ankle_RR = pitch_foot - pitch_tibia; the
ankle is unchanged). The bar joints (rr1/rr2 crank and elbow) follow from the inverse kinematics of the linkage so the
closed loops hold in every frame (the env starts episodes from reference frames). The root is the reference root with
a constant height and pitch so that the lowest stance position of the front and of the hind feet is `--stance-z` (foot
sphere touching the floor; the RRR reference's front feet hover ~6 cm, which made the RR agent sink its trunk), and
its forward travel is scaled to the no-slip speed of the RR legs (`--root-speed`, cyclic.no_slip_root_speed; the
reference's 1.40 m/s is the sheep's speed, far beyond what the leg sweep supports).
"""
import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np
import pandas as pd

from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData

from simulation.config import paths

sys.path.insert(0, str(paths.PROJECT_ROOT.parent))          # repo root: geometry_opt
from geometry_opt import linkage as LK                       # noqa: E402
from geometry_opt import make_rr_xml                         # noqa: E402

LEGS = ("fr", "fl", "br", "bl")


def _pitch(R):
    """Rotation angle about +y of a rotation matrix."""
    return np.arctan2(R[0, 2], R[0, 0])


def _wrap_near(x, ref):
    return ref + (x - ref + np.pi) % (2 * np.pi) - np.pi


def _bar_angles(row, cfg, knee, ankle):
    """rr1/rr2 crank and elbow joint values for arrays of knee and ankle (plane convention of make_rr_xml)."""
    th1, th2 = knee - np.pi / 2, knee - ankle - np.pi / 2
    R1, R2, f1, f2 = row["r1"], row["r2"], row["f1"], row["f2"]
    l1, l2, l3, l4 = row["l1"], row["l2"], row["l3"], cfg["l4"]
    th_ref = cfg["theta"]
    a3 = np.radians(row["a3_deg"])
    prob = dict(R1=R1, R2=R2, ref=th_ref, p=dict(l3_angle=None))
    b1, b2 = row["rr1_branch"], LK._branch_rr2(prob, l3, l4, f2, a3)
    B, O = np.zeros(2), cfg["O"]
    Q1 = O + f1 * R1 * LK.u(th1)[..., :]
    Q2 = O + R1 * LK.u(th1) + f2 * R2 * LK.u(th2)
    K1 = LK._ik(B, Q1, l1, l2, b1)
    K2 = LK._ik(O, Q2, l3, l4, b2)
    ang = lambda v: np.arctan2(v[:, 1], v[:, 0])
    j1, e1 = ang(K1 - B), ang(Q1 - K1) - ang(K1 - B)
    j2, e2 = ang(K2 - O), ang(Q2 - K2) - ang(K2 - O)
    return (_wrap_near(j1, cfg["l1"]), _wrap_near(e1, cfg["l2"] - cfg["l1"]),
            _wrap_near(j2, cfg["l3"]), _wrap_near(e2, cfg["l4_angle"] - cfg["l3"]))


def _cycle_period(x, lo=80, hi=200):
    x = x - x.mean()
    ac = np.correlate(x[:2000], x[:2000], "full")[len(x[:2000]) - 1:]
    return int(np.argmax(ac[lo:hi]) + lo)


def flatten_double_dip(model, leg, knee, ankle, blend_frames=8):
    """Remove the re-rise of a foot's descent to stance (the hind feet of the reference come down to ~2.7 cm above
    their stance height, lift ~1.7 cm and only then land; the agents copied it as a double knee peak / micro-hop).

    Only the bump is changed. In the torso frame (root fixed), on one gait cycle (the clip is a tiled, exactly
    periodic cycle), the foot path between the first low point of the descent (i1, a local minimum: zero vertical
    velocity) and blend_frames after the point where the original path comes back below that height (i2) is replaced
    by a cubic Hermite in height: zero velocity at i1, the original height and vertical velocity at i2. The horizontal
    position is kept everywhere and every other frame of the cycle is unchanged. Knee and ankle on the bump follow by
    inverse kinematics of the foot site (Gauss-Newton on MuJoCo's site Jacobian). Returns the new knee / ankle arrays
    (whole clip) and a summary."""
    d = mujoco.MjData(model)
    qk, qa = model.jnt_qposadr[model.joint(f"{leg}_knee_joint").id], model.jnt_qposadr[model.joint(f"{leg}_ankle_joint").id]
    vk, va = model.jnt_dofadr[model.joint(f"{leg}_knee_joint").id], model.jnt_dofadr[model.joint(f"{leg}_ankle_joint").id]
    site = model.site(f"{leg}_foot_mimic").id
    P = _cycle_period(knee)
    assert np.abs(knee[P:2 * P] - knee[:P]).max() < 1e-6, "reference joints are not periodic"

    def foot(k, a):
        d.qpos[:] = model.qpos0
        d.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
        d.qpos[qk], d.qpos[qa] = k, a
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)                                      # needed by mj_jacSite
        return d.site_xpos[site][[0, 2]].copy()

    k0, a0 = knee[:P].copy(), ankle[:P].copy()
    pos = np.array([foot(k, a) for k, a in zip(k0, a0)])
    z = pos[:, 1]
    i_min, i_apex = int(np.argmin(z)), int(np.argmax(z))
    seg = (i_apex + np.arange((i_min - i_apex) % P + 1)) % P          # apex -> stance minimum (circular)
    zs = z[seg]
    run = np.minimum.accumulate(zs)
    bump = np.nonzero(zs > run + 5e-3)[0]      # frames > 5 mm above the lowest height so far (the tiled cycle
                                               # has a 4 mm seam between its last and first frame)
    if len(bump) == 0:
        return knee, ankle, dict(rerise_mm=0.0, dknee_deg=0.0, dankle_deg=0.0, period=P)
    rerise = float(np.max(zs - run))
    j1 = bump[0] - 1 - int(np.argmin(zs[:bump[0]][::-1]))              # first low point of the descent (last frame at it)
    dz = np.gradient(zs)                                               # per frame
    for extra in range(blend_frames, len(seg)):                       # shortest monotone blend
        j2 = min(bump[-1] + 1 + extra, len(seg) - 1)
        h = np.arange(j2 - j1 + 1) / (j2 - j1)
        z1, z2, v2 = zs[j1], zs[j2], dz[j2] * (j2 - j1)
        h00, h01, h11 = 2 * h**3 - 3 * h**2 + 1, -2 * h**3 + 3 * h**2, h**3 - h**2
        znew = z1 * h00 + z2 * h01 + v2 * h11                          # Hermite: slope 0 at j1, original at j2
        if np.all(np.diff(znew) <= 1e-9):
            break
    assert np.all(np.diff(znew) <= 1e-9), f"{leg}: no monotone blend found"
    frames = seg[j1:j2 + 1]
    target = pos.copy()
    target[frames, 1] = znew
    k1, a1 = k0.copy(), a0.copy()
    jac = np.zeros((3, model.nv))
    for i in frames:
        q = np.array([k1[i], a1[i]])
        for _ in range(50):
            e = target[i] - foot(*q)
            if np.linalg.norm(e) < 1e-7:
                break
            mujoco.mj_jacSite(model, d, jac, None, site)
            J = jac[[0, 2]][:, [vk, va]]
            q = q + np.linalg.solve(J.T @ J + 1e-8 * np.eye(2), J.T @ e)
        assert np.linalg.norm(e) < 1e-5, f"{leg}: foot IK did not converge at frame {i}"
        k1[i], a1[i] = q
    for nm, v in (("knee", k1), ("ankle", a1)):
        lo, hi = model.jnt_range[model.joint(f"{leg}_{nm}_joint").id]
        assert lo <= v.min() and v.max() <= hi, f"{leg} {nm} leaves its range after flattening"
    idx = np.arange(len(knee)) % P
    return k1[idx], a1[idx], dict(rerise_mm=rerise * 1000, dknee_deg=float(np.degrees(np.abs(k1 - k0).max())),
                                  dankle_deg=float(np.degrees(np.abs(a1 - a0).max())), period=P,
                                  frames=(int(frames[0]), int(frames[-1])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="v140", help="linkage design (geometry_opt/results/linkage_<policy>.csv)")
    ap.add_argument("--stance-z", type=float, default=0.027,
                    help="height [m] of the foot site at the lowest stance point (foot sphere radius 0.025 = contact)")
    ap.add_argument("--root-speed", default="no_slip", help="no_slip | keep | <m/s>: forward speed of the reference")
    ap.add_argument("--out", default=str(paths.TRAJ_ADAPTED_RR), help="output .npz")
    ap.add_argument("--keep-hind-double-dip", action="store_true",
                    help="keep the hind feet's double-dip descent of the reference (see flatten_double_dip)")
    args = ap.parse_args(argv)

    ref = np.load(paths.TRAJ_ADAPTED, allow_pickle=True)
    q_o, freq = ref["qpos"], float(ref["frequency"])
    n = q_o.shape[0]
    mo = mujoco.MjModel.from_xml_path(str(paths.XML_PATH)); do = mujoco.MjData(mo)
    mr = mujoco.MjModel.from_xml_path(str(paths.XML_PATH_RR)); dr = mujoco.MjData(mr)
    bid = lambda m, nme: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nme)
    jname = lambda i: mujoco.mj_id2name(mr, mujoco.mjtObj.mjOBJ_JOINT, i)

    # --- knee / ankle of the RR model from the orientation of tibia and metatarsus in the reference -----------
    knee = {l: np.zeros(n) for l in LEGS}; ankle = {l: np.zeros(n) for l in LEGS}
    for i in range(n):
        do.qpos[:] = q_o[i]; mujoco.mj_kinematics(mo, do)
        Rb = do.xmat[bid(mo, "base")].reshape(3, 3)
        for l in LEGS:
            pt = _pitch(Rb.T @ do.xmat[bid(mo, l + "_lower")].reshape(3, 3))
            pf = _pitch(Rb.T @ do.xmat[bid(mo, l + "_foot")].reshape(3, 3))
            knee[l][i], ankle[l][i] = -pt, pf - pt
    for l in LEGS:
        knee[l], ankle[l] = np.unwrap(knee[l]), np.unwrap(ankle[l])
    if not args.keep_hind_double_dip:
        for l in ("br", "bl"):
            knee[l], ankle[l], info = flatten_double_dip(mr, l, knee[l], ankle[l])
            print(f"{l}: double dip flattened (re-rise {info['rerise_mm']:.1f} mm removed, cycle frames "
                  f"{info.get('frames')}), knee / ankle changed by max {info['dknee_deg']:.1f} / {info['dankle_deg']:.1f} deg")

    # --- bar joints from the inverse kinematics of the linkage -----------------------------------------------
    rows = make_rr_xml._leg_rows(args.policy)
    cfgs = {l: LK.reference_configuration(r) for l, r in rows.items()}
    qpos = np.zeros((n, mr.nq))
    qpos[:, :7] = q_o[:, :7]
    for k in range(mr.njnt):
        name = jname(k)
        if name == "root":
            continue
        l, what = name[:2], name[3:]
        adr = mr.jnt_qposadr[k]
        if what == "knee_joint":
            qpos[:, adr] = knee[l]
        elif what == "ankle_joint":
            qpos[:, adr] = ankle[l]
    for l in LEGS:
        j1, e1, j2, e2 = _bar_angles(rows[l], cfgs[l], knee[l], ankle[l])
        for nme, v in (("rr1_joint", j1), ("rr1_elbow", e1), ("rr2_joint", j2), ("rr2_elbow", e2)):
            qpos[:, mr.jnt_qposadr[mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_JOINT, f"{l}_{nme}")]] = v

    # --- root height and pitch so that the stance feet touch the ground as in the reference ------------------
    feet = [mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_SITE, l + "_foot_mimic") for l in LEGS]
    feet_o = [mujoco.mj_name2id(mo, mujoco.mjtObj.mjOBJ_SITE, l + "_foot_mimic") for l in LEGS]
    zr, zo, xr = np.zeros((n, 4)), np.zeros((n, 4)), np.zeros((n, 4))
    for i in range(n):
        dr.qpos[:] = qpos[i]; mujoco.mj_kinematics(mr, dr)
        do.qpos[:] = q_o[i]; mujoco.mj_kinematics(mo, do)
        zr[i], xr[i] = dr.site_xpos[feet, 2], dr.site_xpos[feet, 0] - dr.qpos[0]
        zo[i] = do.site_xpos[feet_o, 2]
    front, rear = [0, 1], [2, 3]
    target = {"front": args.stance_z, "rear": args.stance_z}
    now = {"front": zr[:, front].min(), "rear": zr[:, rear].min()}
    xf, xb = xr[:, front].mean(), xr[:, rear].mean()
    # z' = z + dz - x sin(pitch): solve dz and pitch for both stance minima
    A = np.array([[1.0, -xf], [1.0, -xb]])
    dz, pitch = np.linalg.solve(A, [target["front"] - now["front"], target["rear"] - now["rear"]])
    print(f"root: dz = {dz * 1000:+.1f} mm, nose-down pitch = {np.degrees(pitch):.2f} deg")
    qy = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])          # rotation about +y
    w1, x1, y1, z1 = qy[0], qy[1], qy[2], qy[3]
    q0 = qpos[:, 3:7]
    qpos[:, 3:7] = np.stack([w1 * q0[:, 0] - x1 * q0[:, 1] - y1 * q0[:, 2] - z1 * q0[:, 3],
                             w1 * q0[:, 1] + x1 * q0[:, 0] + y1 * q0[:, 3] - z1 * q0[:, 2],
                             w1 * q0[:, 2] - x1 * q0[:, 3] + y1 * q0[:, 0] + z1 * q0[:, 1],
                             w1 * q0[:, 3] + x1 * q0[:, 2] - y1 * q0[:, 1] + z1 * q0[:, 0]], axis=1)
    qpos[:, 2] += dz

    # --- velocities: root as in the reference, joints by differentiating the angles ---------------------------
    qvel = np.zeros((n, mr.nv))
    qvel[:, :6] = ref["qvel"][:, :6]

    # --- forward speed ------------------------------------------------------------------------------------------
    v_ref = float(qvel[:, 0].mean())
    if args.root_speed != "keep":
        if args.root_speed == "no_slip":
            from simulation.mocap.cyclic import no_slip_root_speed
            v_new, ns = no_slip_root_speed(mr, qpos, freq)
            print(f"no-slip speed of the RR legs: {v_new:.3f} m/s (foot x sweep {np.round(ns['sweep_x'], 3)} m)")
        else:
            v_new = float(args.root_speed)
        k = v_new / v_ref
        qpos[:, 0:2] = qpos[0, 0:2] + k * (qpos[:, 0:2] - qpos[0, 0:2])
        qvel[:, 0:2] *= k
        print(f"root speed {v_ref:.3f} -> {v_new:.3f} m/s")
    for k in range(1, mr.njnt):
        qvel[:, mr.jnt_dofadr[k]] = np.gradient(qpos[:, mr.jnt_qposadr[k]], 1.0 / freq)

    # --- forward kinematics data for loco-mujoco ----------------------------------------------------------------
    site_names = [mujoco.mj_id2name(mr, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(mr.nsite)]
    site_names = [s for s in site_names if s and s.endswith("_mimic")]
    site_ids = [mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_SITE, s) for s in site_names]
    out = dict(xpos=np.zeros((n, mr.nbody, 3)), xquat=np.zeros((n, mr.nbody, 4)), cvel=np.zeros((n, mr.nbody, 6)),
               subtree_com=np.zeros((n, mr.nbody, 3)), site_xpos=np.zeros((n, len(site_ids), 3)),
               site_xmat=np.zeros((n, len(site_ids), 9)))
    for i in range(n):
        dr.qpos[:], dr.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_kinematics(mr, dr); mujoco.mj_comPos(mr, dr); mujoco.mj_comVel(mr, dr); mujoco.mj_crb(mr, dr)
        out["xpos"][i], out["xquat"][i], out["cvel"][i], out["subtree_com"][i] = dr.xpos, dr.xquat, dr.cvel, dr.subtree_com
        out["site_xpos"][i] = dr.site_xpos[site_ids]
        out["site_xmat"][i] = dr.site_xmat[site_ids].reshape(len(site_ids), 9)

    # closure of the loops at a few frames (should be ~0)
    viol = 0.0
    for i in range(0, n, 97):
        dr.qpos[:] = qpos[i]; mujoco.mj_forward(mr, dr)
        viol = max(viol, float(np.abs(dr.efc_pos[:dr.ne]).max()) if dr.ne else 0.0)
    print(f"max equality violation over sampled frames: {viol:.2e} m")

    joint_names = [jname(k) for k in range(mr.njnt)]
    info = TrajectoryInfo(joint_names,
                          model=TrajectoryModel(mr.njnt, jnp.array(mr.jnt_type), nsite=mr.nsite,
                                                site_bodyid=jnp.zeros((0,), dtype=jnp.int32),
                                                site_pos=jnp.zeros((0, 3)), site_quat=jnp.zeros((0, 4))),
                          frequency=freq, site_names=site_names)
    data = TrajectoryData(jnp.array(qpos), jnp.array(qvel), xpos=jnp.array(out["xpos"]), xquat=jnp.array(out["xquat"]),
                          cvel=jnp.array(out["cvel"]), subtree_com=jnp.array(out["subtree_com"]),
                          site_xpos=jnp.array(out["site_xpos"]), site_xmat=jnp.array(out["site_xmat"]),
                          split_points=jnp.array(ref["split_points"]))
    out_path = Path(args.out)
    Trajectory(info, data).save(out_path.with_suffix(""))
    print(f"saved {out_path}")
    for l in LEGS[::2]:
        k = mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_JOINT, f"{l}_knee_joint")
        print(l, "knee range [deg]", np.degrees(knee[l].min()).round(1), np.degrees(knee[l].max()).round(1),
              "model range", np.degrees(mr.jnt_range[k]).round(1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
