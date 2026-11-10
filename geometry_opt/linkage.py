"""
Transmission linkage of a 2R leg (tibia r1 + metatarsus r2), driven from the torso by two RR bar mechanisms.
Port of `MecanismoV2.ipynb` (Tesis/Python/leg_transmission).

Planar model, all angles absolute (measured from the mechanism's +x axis), lengths in metres:

    B  = actuator of RR 1, at the origin.        O = pivot of r1 = (dx, -dy) from B.
    RR 1 (l1, l2): base B, tip Q1 = O + f1 r1.   RR 2 (l3, l4): base O, tip Q2 = O + r1 + f2 r2.
    P  = O + r1 + r2 (end of the leg).
    A RR with bars a, b reaches a point at distance D iff |a - b| <= D <= (a + b) / (1 + reach_margin).

l4 is not free: it is the distance from the tip of l3 (at absolute angle a3 from O) to the point f2 r2 in the
reference pose (centre of the range of motion). The unknowns are x = (l1, l2, l3, dx, dy, a3).

For each (f1, f2) of the sweep, differential evolution minimises

    w_l4 l4/0.1 + w_length (l1+l2+l3)/0.3 + w_offset (rise of dx+dy over its minimum)/0.05
      + w_range (1 - fraction of the extended range that both RR reach without overlap)
      + w_torque mean_k(|tau_act_k|max / |tau_joint_k|max)      (only with the torques of the gait: tau_ref)
      + 50 (1 - fraction of the nominal range reached without overlap)

The torque term is the amplification of the joint torque through the linkage (1 = direct drive, 5 = the actuator
needs five times the torque of the knee / ankle it replaces), computed with the torque of the gait on a subset of
poses (the extremes of both joints plus an even sample); without it, short compact bars with a poor lever arm win.

Overlap: every pair of bars that does not share a joint must stay farther apart than `clearance` at every
pose (bars are segments without width; the torso and the fixed joints are not checked).

The range of motion comes from the recorded gait: the orientation of the tibia and the metatarsus with
respect to the torso, from MuJoCo forward kinematics. It only involves those two links, so it does not
change when the thigh is removed (RRR -> RR model) as long as they keep their position and length.

This module needs only numpy/scipy (mujoco and matplotlib are imported lazily), so the worker processes of
the sweep start fast.
"""
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import differential_evolution

from . import config as C

u = lambda a: np.stack([np.cos(a), np.sin(a)], axis=-1)
LINK_LABELS = ("r1", "r2", "l1", "l2", "l3", "l4")
LINK_COLORS = ("tab:blue", "tab:blue", "tab:orange", "tab:orange", "tab:red", "tab:red")
# Bars 0 r1, 1 r2, 2 l1, 3 l2, 4 l3, 5 l4. Pairs that share a joint are not checked
# (r1-r2, l1-l2, l3-l4, r1-l3 at O, r2-l4 at Q2, r1-l2 at Q1).
PAIRS = [(0, 2), (0, 5), (1, 2), (1, 3), (1, 4), (2, 4), (2, 5), (3, 4), (3, 5)]


# ------------------------------------------------------------------------------------------------
# Inputs from the MuJoCo model and the recorded gait
# ------------------------------------------------------------------------------------------------
def _leg_capsules(model, leg):
    import mujoco
    out = []
    for seg in ("lower", "foot"):
        b = model.body(f"{leg}_{seg}").id
        gs = [g for g in range(model.ngeom)
              if model.geom_bodyid[g] == b and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]
        assert len(gs) == 1, f"{leg}_{seg}: expected one capsule"
        out.append((b, gs[0]))
    return out


def leg_lengths(model, leg):
    """(r1, r2) = lengths [m] of the tibia (`<leg>_lower`) and the metatarsus (`<leg>_foot`) capsules."""
    return tuple(2.0 * float(model.geom_size[g, 1]) for _, g in _leg_capsules(model, leg))


def plane_angles(model, qpos, leg, x_forward=True):
    """Absolute plane angles (N, 2) of tibia and metatarsus for every gait sample.

    Forward kinematics: the capsule axis (from joint to distal end) expressed in the torso frame gives the
    sagittal pitch phi (0 = hanging down, + = backwards, direction (x fwd, z up) = (-sin phi, -cos phi)); the
    mechanism angle is -pi/2 - phi (x_forward) or -pi/2 + phi. On the RRR model this equals
    phi_tibia = hip - knee and phi_metatarsus = hip - knee + ankle."""
    import mujoco
    data = mujoco.MjData(model)
    base = model.body("base").id
    caps = _leg_capsules(model, leg)
    axes = []
    for _, g in caps:
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        ax = R.reshape(3, 3)[:, 2]                          # capsule axis in its body frame
        axes.append(ax if model.geom_pos[g] @ ax >= 0 else -ax)   # oriented from the joint to the distal end
    phi = np.zeros((len(qpos), 2))
    for s, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        Rb = data.xmat[base].reshape(3, 3)
        for k, ((b, _), ax) in enumerate(zip(caps, axes)):
            d = Rb.T @ (data.xmat[b].reshape(3, 3) @ ax)
            phi[s, k] = np.arctan2(-d[0], -d[2])
    return -np.pi / 2 + (-1 if x_forward else 1) * phi


# ------------------------------------------------------------------------------------------------
# Problem of one leg
# ------------------------------------------------------------------------------------------------
def _grid(rng, n):
    a, b = np.meshgrid(*[np.linspace(*r, n) for r in rng], indexing="ij")
    return a.ravel(), b.ravel()


def make_problem(name, R1, R2, ang, l_max, params=None, tau_joint=None):
    """Problem of one leg. `ang` (N, 2): absolute plane angles of tibia and metatarsus over the gait;
    `tau_joint` (N, 2): knee and ankle joint torque of the same samples [N·m] (enables the torque term)."""
    p = params or C.linkage_params()
    ang = np.asarray(ang, dtype=float)
    ext = np.radians(p["range_margin_deg"])
    rng_nom = [(float(ang[:, k].min()), float(ang[:, k].max())) for k in range(2)]
    rng_ext = [(a - ext, b + ext) for a, b in rng_nom]
    tau_ref = None
    if tau_joint is not None:
        tau_joint = np.asarray(tau_joint, dtype=float)
        k = p["torque_samples"]
        idx = set(np.linspace(0, len(ang) - 1, max(k // 2, 2)).astype(int).tolist())
        for j in range(2):
            idx |= set(np.argsort(-np.abs(tau_joint[:, j]))[: max(k // 4, 1)].tolist())
        idx = sorted(idx)
        tau_ref = dict(theta=ang[idx], tau=tau_joint[idx], peak=np.abs(tau_joint).max(0))
    return dict(name=name, R1=float(R1), R2=float(R2), rng_nom=rng_nom, rng_ext=rng_ext, tau_ref=tau_ref,
                ref=np.array([np.mean(r) for r in rng_nom]),              # reference pose: centre of the range
                g_nom=_grid(rng_nom, p["grid"][0]), g_ext=_grid(rng_ext, p["grid"][1]),
                bounds=[(0.01, l_max)] * 3 + [(p["dx_min"], p["dx_min"] + p["d_extra"]),
                                              (p["dy_min"], p["dy_min"] + p["d_extra"]), (-np.pi, np.pi)],
                p=p)


# ------------------------------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------------------------------
def _point_seg(p, a, b):
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=-1)


def _cross(a, b, c, d):
    cr = lambda p, q, r: (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])
    return (cr(a, b, c) * cr(a, b, d) < 0) & (cr(c, d, a) * cr(c, d, b) < 0)


def _seg_dist(a, b, c, d):
    m = np.minimum.reduce([_point_seg(a, c, d), _point_seg(b, c, d), _point_seg(c, a, b), _point_seg(d, a, b)])
    return np.where(_cross(a, b, c, d), 0.0, m)


def _ik(base, tip, a, b, branch):
    v = tip - base
    D = np.linalg.norm(v, axis=-1)
    c = np.clip((a ** 2 + D ** 2 - b ** 2) / (2 * a * np.maximum(D, 1e-9)), -1, 1)
    return base + a * u(np.arctan2(v[..., 1], v[..., 0]) + branch * np.arccos(c))


def _a3(prob, x):
    return x[5] if prob["p"]["l3_angle"] is None else prob["p"]["l3_angle"]


def l4_of(prob, l3, f2, a3):
    """l4: distance from the tip of l3 (angle a3 from O) to the point f2 of r2, in the reference pose."""
    ref = prob["ref"]
    return float(np.linalg.norm(l3 * u(a3) - (prob["R1"] * u(ref[0]) + f2 * prob["R2"] * u(ref[1]))))


def _branch_rr2(prob, l3, l4, f2, a3):
    # elbow branch of RR 2 that puts l3 at a3 in the reference pose (kept over the whole range)
    ref = prob["ref"]
    v = prob["R1"] * u(ref[0]) + f2 * prob["R2"] * u(ref[1])
    D = np.linalg.norm(v)
    c = np.clip((l3 ** 2 + D ** 2 - l4 ** 2) / (2 * l3 * max(D, 1e-9)), -1, 1)
    a0 = np.arctan2(v[1], v[0])
    w = lambda t: abs(np.angle(np.exp(1j * t)))
    return +1 if w(a0 + np.arccos(c) - a3) <= w(a0 - np.arccos(c) - a3) else -1


def evaluate(prob, x, f1, f2, grid, branch1, detail=False):
    """Poses of `grid` that both RR reach without overlap, for x = (l1, l2, l3, dx, dy, a3) and the elbow
    branch `branch1` (+1/-1) of RR 1. Returns (ok mask, l4); with detail=True: (reach, free, l4, bars)
    where bars = [(start, end)] x 6 in the order of LINK_LABELS."""
    R1, R2, p = prob["R1"], prob["R2"], prob["p"]
    l1, l2, l3, dx, dy = x[:5]
    a3 = _a3(prob, x)
    l4 = l4_of(prob, l3, f2, a3)
    th1, th2 = grid
    n = len(th1)
    O = np.broadcast_to(np.array([dx, -dy]), (n, 2))
    B = np.zeros((n, 2))
    J = O + R1 * u(th1)
    P = J + R2 * u(th2)
    Q1 = O + f1 * R1 * u(th1)
    Q2 = J + f2 * R2 * u(th2)
    D1 = np.linalg.norm(Q1, axis=1)
    D2 = np.linalg.norm(Q2 - O, axis=1)
    k = 1 + p["reach_margin"]
    reach = ((D1 >= abs(l1 - l2)) & (D1 <= (l1 + l2) / k) & (D2 >= abs(l3 - l4)) & (D2 <= (l3 + l4) / k))
    K1 = _ik(B, Q1, l1, l2, branch1)
    K2 = _ik(O, Q2, l3, l4, _branch_rr2(prob, l3, l4, f2, a3))
    bars = [(O, J), (J, P), (B, K1), (K1, Q1), (O, K2), (K2, Q2)]
    A0 = np.stack([b[0] for b in bars])
    A1 = np.stack([b[1] for b in bars])
    ii, jj = [list(t) for t in zip(*PAIRS)]
    d = _seg_dist(A0[ii].reshape(-1, 2), A1[ii].reshape(-1, 2), A0[jj].reshape(-1, 2), A1[jj].reshape(-1, 2))
    free = (d.reshape(len(PAIRS), n) > p["clearance"]).all(axis=0)
    if detail:
        return reach, free, l4, bars
    return reach & free, l4


def _act_angles(prob, x, f1, f2, b1, theta):
    """Absolute angle (N, 2) of the driven bars l1 (RR 1 actuator, at B) and l3 (RR 2 actuator, at O) for the
    plane angles theta (N, 2) of tibia and metatarsus."""
    R1, R2 = prob["R1"], prob["R2"]
    l1, l2, l3, dx, dy = x[:5]
    a3 = _a3(prob, x)
    l4 = l4_of(prob, l3, f2, a3)
    th1, th2 = theta[:, 0], theta[:, 1]
    O = np.broadcast_to(np.array([dx, -dy]), (len(th1), 2))
    Q1 = O + f1 * R1 * u(th1)
    Q2 = O + R1 * u(th1) + f2 * R2 * u(th2)
    K1 = _ik(np.zeros_like(O), Q1, l1, l2, b1)
    K2 = _ik(O, Q2, l3, l4, _branch_rr2(prob, l3, l4, f2, a3))
    return np.stack([np.arctan2(K1[:, 1], K1[:, 0]), np.arctan2((K2 - O)[:, 1], (K2 - O)[:, 0])], axis=1)


def _act_jacobian(prob, x, f1, f2, b1, theta, eps=1e-6):
    G = np.zeros((len(theta), 2, 2))
    for j in range(2):
        dp, dm = theta.copy(), theta.copy()
        dp[:, j] += eps
        dm[:, j] -= eps
        G[:, :, j] = np.angle(np.exp(1j * (_act_angles(prob, x, f1, f2, b1, dp)
                                           - _act_angles(prob, x, f1, f2, b1, dm)))) / (2 * eps)
    return G


RATIO_CAP = 20.0     # cap of the torque amplification used in the cost (also for singular poses)


def torque_ratio(prob, x, f1, f2, b1):
    """Mean over the two actuators of |tau_act|max / |tau_joint|max for the torque of the gait (nan without tau_ref)."""
    t = prob["tau_ref"]
    if t is None:
        return np.nan
    A = joint_to_plane_matrix(prob["p"]["x_forward"])
    G = _act_jacobian(prob, x, f1, f2, b1, t["theta"])
    try:
        tau_theta = np.linalg.solve(A.T, t["tau"].T).T
        tau_a = np.linalg.solve(np.transpose(G, (0, 2, 1)), tau_theta[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        return RATIO_CAP
    r = np.abs(tau_a).max(0) / np.maximum(t["peak"], 1e-9)
    return float(min(np.mean(r), RATIO_CAP)) if np.all(np.isfinite(r)) else RATIO_CAP


def cost(x, prob, f1, f2):
    p = prob["p"]
    w = p["weights"]
    base = (w["l4"] * l4_of(prob, x[2], f2, _a3(prob, x)) / 0.1 + w["length"] * (x[0] + x[1] + x[2]) / 0.3
            + w["offset"] * (x[3] - p["dx_min"] + x[4] - p["dy_min"]) / 0.05)
    best = np.inf
    for b1 in (+1, -1):
        ok_n, _ = evaluate(prob, x, f1, f2, prob["g_nom"], b1)
        if ok_n.all():                      # the extended range is only evaluated once the nominal one is covered
            ok_e, _ = evaluate(prob, x, f1, f2, prob["g_ext"], b1)
            c = base + w["range"] * (1 - ok_e.mean())
            if prob["tau_ref"] is not None:
                c += w["torque"] * torque_ratio(prob, x, f1, f2, b1)
        else:
            c = base + w["range"] + 50 * (1 - ok_n.mean())
            if prob["tau_ref"] is not None:
                c += w["torque"] * RATIO_CAP          # a pose set that is not covered never beats a covered one
        best = min(best, c)
    return best


# ------------------------------------------------------------------------------------------------
# Sweep and selection
# ------------------------------------------------------------------------------------------------
def _solve_cell(args):
    prob, f1, f2 = args
    r = differential_evolution(cost, prob["bounds"], args=(prob, f1, f2), polish=False, **prob["p"]["de"])
    opts = []
    for b1 in (+1, -1):
        ok_n, l4 = evaluate(prob, r.x, f1, f2, prob["g_nom"], b1)
        ok_e, _ = evaluate(prob, r.x, f1, f2, prob["g_ext"], b1)
        opts.append((50 * ok_n.mean() + ok_e.mean(), b1, ok_n.mean(), ok_e.mean(), l4))
    _, b1, cn, ce, l4 = max(opts)
    return r.x, l4, b1, cn, ce, r.fun, torque_ratio(prob, r.x, f1, f2, b1)


def _signature(prob):
    blob = json.dumps(dict(R1=prob["R1"], R2=prob["R2"], rng=prob["rng_nom"], p=prob["p"]), sort_keys=True,
                      default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    h = hashlib.sha1(blob.encode())
    if prob["tau_ref"] is not None:
        for k in ("theta", "tau", "peak"):
            h.update(np.ascontiguousarray(prob["tau_ref"][k]).tobytes())
    return h.hexdigest()


def sweep(prob, workers=None, cache=None, log=print, reload=True):
    """Optimise every (f1, f2) cell. With `cache` (a .npz path) the sweep is saved there, and reused when the
    inputs match (unless reload=False)."""
    sig = _signature(prob)
    if cache is not None and reload and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if str(z["signature"]) == sig:
            log(f"    [{prob['name']}] linkage sweep loaded from {cache.name}")
            return {k: z[k] for k in ("X", "L4", "BRANCH", "COV_N", "COV_E", "COST", "RATIO")}
    f_vals = prob["p"]["f_values"]
    cells = [(prob, f1, f2) for f1 in f_vals for f2 in f_vals]
    workers = workers or C.LINKAGE_WORKERS or mp.cpu_count()
    if workers == 1:
        out = [_solve_cell(c) for c in cells]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(cells)), mp_context=mp.get_context("spawn")) as ex:
            out = list(ex.map(_solve_cell, cells))
    n = len(f_vals)
    res = dict(X=np.array([o[0] for o in out]).reshape(n, n, 6), L4=np.array([o[1] for o in out]).reshape(n, n),
               BRANCH=np.array([o[2] for o in out]).reshape(n, n), COV_N=np.array([o[3] for o in out]).reshape(n, n),
               COV_E=np.array([o[4] for o in out]).reshape(n, n), COST=np.array([o[5] for o in out]).reshape(n, n),
               RATIO=np.array([o[6] for o in out]).reshape(n, n))
    if cache is not None:
        np.savez_compressed(cache, signature=sig, **res)
    return res


def select_best(prob, sw):
    f = np.array(prob["p"]["f_values"])
    ok = (f[:, None] >= prob["p"]["f_min"]) & (f[None, :] >= prob["p"]["f_min"]) & (sw["COV_N"] >= 1.0)
    if not ok.any():
        raise RuntimeError(f"[{prob['name']}] no (f1, f2) covers the nominal range without overlap")
    i, j = np.unravel_index(np.where(ok, sw["COST"], np.inf).argmin(), ok.shape)
    x = sw["X"][i, j].copy()
    if prob["p"]["l3_angle"] is not None:
        x[5] = prob["p"]["l3_angle"]
    return dict(f1=float(f[i]), f2=float(f[j]), x=x, branch=int(sw["BRANCH"][i, j]), l4=float(sw["L4"][i, j]),
                cov_nom=float(sw["COV_N"][i, j]), cov_ext=float(sw["COV_E"][i, j]), ratio=float(sw["RATIO"][i, j]),
                cell=(int(i), int(j)))


def design_leg(prob, workers=None, cache=None, log=print, reload=True):
    """Sweep + selection for one leg. Returns dict(prob, sweep, best)."""
    sw = sweep(prob, workers, cache, log, reload)
    best = select_best(prob, sw)
    l1, l2, l3, dx, dy, a3 = best["x"]
    log(f"    [{prob['name']}] f1={best['f1']:.2f} f2={best['f2']:.2f} | l1={l1*1000:.1f} l2={l2*1000:.1f} "
        f"l3={l3*1000:.1f} l4={best['l4']*1000:.1f} mm | dx={dx*1000:.1f} dy={dy*1000:.1f} mm | "
        f"a3={np.degrees(a3):.0f} deg | extended range {100*best['cov_ext']:.0f} %"
        + (f" | torque amplification {best['ratio']:.2f}x" if np.isfinite(best["ratio"]) else ""))
    return dict(prob=prob, sweep=sw, best=best)


# ------------------------------------------------------------------------------------------------
# Actuators of the linkage: torque and speed requirements through the mechanism
# ------------------------------------------------------------------------------------------------
def actuator_angles(res, theta):
    """Absolute angle (N, 2) of the two driven bars for the plane angles theta (N, 2) of tibia and metatarsus:
    column 0 = l1 (actuator of RR 1, at B), column 1 = l3 (actuator of RR 2, at O)."""
    b = res["best"]
    return _act_angles(res["prob"], b["x"], b["f1"], b["f2"], b["branch"], np.asarray(theta, dtype=float))


def actuator_jacobian(res, theta, eps=1e-6):
    """G (N, 2, 2) = d(actuator angles) / d(plane angles of tibia and metatarsus), by central differences.
    RR 1 only sees the tibia (G[:, 0, 1] = 0); RR 2 sees both."""
    b = res["best"]
    return _act_jacobian(res["prob"], b["x"], b["f1"], b["f2"], b["branch"], np.asarray(theta, dtype=float), eps)


def joint_to_plane_matrix(x_forward=True):
    """A = d(plane angles) / d(knee, ankle) with the hip removed (RR model): the tibia pitch is -knee and the
    metatarsus pitch -knee + ankle (MuJoCo convention), and plane angle = -pi/2 -/+ pitch."""
    s = -1.0 if x_forward else 1.0
    return s * np.array([[-1.0, 0.0], [-1.0, 1.0]])


def actuator_loads(res, theta, tau_joint, omega_joint, x_forward=True):
    """Torque and speed of the two actuators of a leg for a gait.

    theta (S, 2): plane angles of tibia/metatarsus; tau_joint, omega_joint (S, 2): knee and ankle joint torque
    [N·m] and speed [rad/s] of the gait. Virtual work: tau_theta = A^-T tau_q, tau_act = G^-T tau_theta and
    omega_act = G A omega_q (power is conserved). Returns (tau_act (S,2), omega_act (S,2), det G (S,))."""
    A = joint_to_plane_matrix(x_forward)
    G = actuator_jacobian(res, theta)
    tau_theta = np.linalg.solve(A.T, np.asarray(tau_joint).T).T
    tau_act = np.linalg.solve(np.transpose(G, (0, 2, 1)), tau_theta[:, :, None])[:, :, 0]
    omega_act = np.einsum("sij,jk,sk->si", G, A, np.asarray(omega_joint))
    return tau_act, omega_act, np.linalg.det(G)


def actuator_requirements(results, loads, joint_names, x_forward=True):
    """Requirements of the actuators of the linkage and of the joints they replace.

    results: design_legs()[0] (with res['theta'] = plane angles of the gait, set by design_legs); loads: the
    inverse_dynamics.Loads of a design; joint_names: model_builder.JOINT_NAMES. Returns a list of dicts, two per
    designed leg (RR 1 <-> knee, RR 2 <-> ankle) with the actuator requirement and that of the joint it replaces."""
    rows = []
    rpm = 60 / (2 * np.pi)
    for leg, res in results.items():
        ik_, ia_ = joint_names.index(f"{leg}_knee"), joint_names.index(f"{leg}_ankle")
        tau_q = loads.tau[:, [ik_, ia_]]
        w_q = loads.omega[:, [ik_, ia_]]
        tau_a, w_a, det = actuator_loads(res, res["theta"], tau_q, w_q, x_forward)
        for k, nm in enumerate(("rr1", "rr2")):
            rows.append(dict(
                actuator=f"{leg}_{nm}", leg=leg, tau_max=float(np.abs(tau_a[:, k]).max()),
                tau_rms=float(np.sqrt((tau_a[:, k] ** 2).mean())), omega_max_rpm=float(np.abs(w_a[:, k]).max() * rpm),
                joint=("knee", "ankle")[k], joint_tau_max=float(np.abs(tau_q[:, k]).max()),
                joint_tau_rms=float(np.sqrt((tau_q[:, k] ** 2).mean())),
                joint_omega_max_rpm=float(np.abs(w_q[:, k]).max() * rpm), min_abs_det_G=float(np.abs(det).min()),
                power_error=float(np.abs((tau_a * w_a).sum(1) - (tau_q * w_q).sum(1)).max())))
    return rows


def reference_configuration(row, x_forward=True):
    """Closed reference pose of a designed leg (centre of its range of motion), from a `summary_row`.

    Everything in the mechanism plane (x forward, y up = torso x, z), angles in rad. Returns the absolute angle
    of the six bars (tibia, metatarsus, l1, l2, l3, l4), the derived l4, the joint values of the RR leg in the
    MuJoCo convention (knee, ankle: tibia pitch = -knee, metatarsus pitch = -knee + ankle) and the closure
    errors of both loops [m] (should be ~0)."""
    assert x_forward, "only x_forward=True is supported (mechanism plane x = torso x, plane y = torso z)"
    R1, R2, f1, f2 = row["r1"], row["r2"], row["f1"], row["f2"]
    l1, l2, l3, dx, dy, a3 = row["l1"], row["l2"], row["l3"], row["dx"], row["dy"], np.radians(row["a3_deg"])
    th = np.radians([0.5 * (row["theta1_min_deg"] + row["theta1_max_deg"]), 0.5 * (row["theta2_min_deg"] + row["theta2_max_deg"])])
    prob = dict(R1=R1, R2=R2, ref=th, p=dict(l3_angle=None))
    l4 = l4_of(prob, l3, f2, a3)
    O, B = np.array([dx, -dy]), np.zeros(2)
    J = O + R1 * u(th[0])
    Q1 = O + f1 * R1 * u(th[0])
    Q2 = J + f2 * R2 * u(th[1])
    K1 = _ik(B[None], Q1[None], l1, l2, row["rr1_branch"])[0]
    K2 = _ik(O[None], Q2[None], l3, l4, _branch_rr2(prob, l3, l4, f2, a3))[0]
    ang = lambda v: float(np.arctan2(v[1], v[0]))
    phi = -(th + np.pi / 2)                                     # pitch of tibia and metatarsus (x_forward)
    return dict(theta=th, l4=l4, tibia=th[0], metatarsus=th[1], l1=ang(K1 - B), l2=ang(Q1 - K1), l3=ang(K2 - O),
                l4_angle=ang(Q2 - K2), knee=float(-phi[0]), ankle=float(phi[1] - phi[0]), B=B, O=O, J=J, K1=K1, K2=K2,
                Q1=Q1, Q2=Q2, closure=(float(abs(np.linalg.norm(Q1 - K1) - l2)), float(abs(np.linalg.norm(Q2 - K2) - l4))))


def design_legs(gait, key, out_dir, legs=None, xml=None, workers=None, use_cache=True, plots=True, log=print,
                loads=None, joint_names=None):
    """Design the linkage of every leg of `legs` (default C.LINKAGE_LEGS) for a recorded gait.

    `gait` is the dict of simulation.record_gait (qpos of the model `xml`, default C.LINKAGE_XML). Writes
    linkage_<key>_<leg>.npz (sweep cache) and, with plots, _maps.png, _design.png and _animation.gif into out_dir.
    With `loads` (inverse_dynamics.Loads of the gait on the thesis model) and `joint_names`
    (model_builder.JOINT_NAMES) the torque of the knee and ankle joints enters the cost (torque amplification).
    Returns (results, rows): results[leg] = dict(prob, sweep, best, trajectory=(reach, free)); rows for a CSV."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(xml or C.LINKAGE_XML))
    p = C.linkage_params()
    n_envs, n_per = int(gait["n_envs"]), int(gait["n_per_env"])
    results, rows = {}, []
    for leg in legs or C.LINKAGE_LEGS:
        R1, R2 = leg_lengths(model, leg)
        ang = plane_angles(model, gait["qpos"], leg, p["x_forward"])
        tau = None if loads is None else loads.tau[:, [joint_names.index(f"{leg}_knee"), joint_names.index(f"{leg}_ankle")]]
        prob = make_problem(leg.upper(), R1, R2, ang, p["l_max"]["front" if leg[0] == "f" else "back"], p, tau)
        res = design_leg(prob, workers, out_dir / f"linkage_{key}_{leg}.npz", log, reload=use_cache)
        res["trajectory"] = check_trajectory(res, ang)
        res["theta"] = ang
        log(f"    [{prob['name']}] gait samples reached / without overlap: "
            f"{100*res['trajectory'][0]:.1f} % / {100*res['trajectory'][1]:.1f} %")
        if plots:
            plot_maps(res, out_dir / f"linkage_{key}_{leg}_maps.png")
            plot_design(res, out_dir / f"linkage_{key}_{leg}_design.png")
            animate(res, ang.reshape(n_per, n_envs, 2)[:, 0], out_dir / f"linkage_{key}_{leg}_animation.gif",
                    dt=float(gait["dt"]))
        results[leg] = res
        rows.append(dict(summary_row(res, key), gait_reached_pct=100 * res["trajectory"][0],
                         gait_no_overlap_pct=100 * res["trajectory"][1]))
    return results, rows


def summary_row(res, model_key=None):
    """One flat row (SI units, angles in deg) for CSV / report."""
    prob, b = res["prob"], res["best"]
    l1, l2, l3, dx, dy, a3 = b["x"]
    row = dict(leg=prob["name"], r1=prob["R1"], r2=prob["R2"], f1=b["f1"], f2=b["f2"], l1=l1, l2=l2, l3=l3, l4=b["l4"],
               dx=dx, dy=dy, a3_deg=float(np.degrees(a3)), rr1_branch=b["branch"],
               nominal_range_pct=100 * b["cov_nom"], extended_range_pct=100 * b["cov_ext"], torque_amplification=b["ratio"],
               theta1_min_deg=float(np.degrees(prob["rng_nom"][0][0])), theta1_max_deg=float(np.degrees(prob["rng_nom"][0][1])),
               theta2_min_deg=float(np.degrees(prob["rng_nom"][1][0])), theta2_max_deg=float(np.degrees(prob["rng_nom"][1][1])))
    return dict(model=model_key, **row) if model_key else row


def check_trajectory(res, ang):
    """Fraction of gait samples (N, 2 plane angles) that both RR reach / that have no overlapping bars."""
    b = res["best"]
    reach, free, _, _ = evaluate(res["prob"], b["x"], b["f1"], b["f2"], (ang[:, 0], ang[:, 1]), b["branch"], detail=True)
    return float(reach.mean()), float(free.mean())


# ------------------------------------------------------------------------------------------------
# Plots (matplotlib imported lazily)
# ------------------------------------------------------------------------------------------------
def plot_maps(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prob, sw, b = res["prob"], res["sweep"], res["best"]
    p = prob["p"]
    f = np.array(p["f_values"])
    st = f[1] - f[0] if len(f) > 1 else 0.1
    lo, hi = f[0] - st / 2, f[-1] + st / 2
    ok = sw["COV_N"] >= 1
    X = sw["X"]
    maps = [("l4 [mm]", np.where(ok, sw["L4"], np.nan) * 1000),
            ("l1 + l2 + l3 [mm]", np.where(ok, X[:, :, :3].sum(2), np.nan) * 1000),
            ("dx + dy above the minimum [mm]", np.where(ok, X[:, :, 3] + X[:, :, 4] - p["dx_min"] - p["dy_min"], np.nan) * 1000),
            ("angle of l3 in the reference pose [deg]", np.where(ok, np.degrees(X[:, :, 5]), np.nan)),
            ("extended range of P reached [%]", sw["COV_E"] * 100),
            ("actuator / joint torque (mean of RR 1, RR 2)", np.where(ok, sw["RATIO"], np.nan)), ("cost", sw["COST"])]
    fig, axs = plt.subplots(2, 4, figsize=(20, 8.5))
    for ax in axs.ravel()[len(maps):]:
        ax.axis("off")
    for ax, (ttl, M) in zip(axs.ravel(), maps):
        im = ax.imshow(M, origin="lower", extent=[lo, hi, lo, hi], aspect="auto", cmap="viridis")
        ax.axvspan(lo, p["f_min"] - st / 2, color="k", alpha=0.15)
        ax.axhspan(lo, p["f_min"] - st / 2, color="k", alpha=0.15)
        ax.plot(b["f2"], b["f1"], "r*", ms=14)
        ax.set_xlabel("f2")
        ax.set_ylabel("f1")
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"Leg {prob['name']}")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_design(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    prob, b = res["prob"], res["best"]
    R1, R2 = prob["R1"], prob["R2"]
    O = np.array([b["x"][3], -b["x"][4]])
    ok_e, _ = evaluate(prob, b["x"], b["f1"], b["f2"], prob["g_ext"], b["branch"])
    Pe = O + R1 * u(prob["g_ext"][0]) + R2 * u(prob["g_ext"][1])
    Pn = O + R1 * u(prob["g_nom"][0]) + R2 * u(prob["g_nom"][1])
    hv = ConvexHull(Pn).vertices

    def pose(t1, t2):
        _, _, _, bars = evaluate(prob, b["x"], b["f1"], b["f2"], (np.array([t1]), np.array([t2])), b["branch"], detail=True)
        return [(s[0], e[0]) for s, e in bars]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(14, 7.5), sharey=True)
    axa.scatter(*Pe[~ok_e].T, s=14, c="lightgray", label="unreachable or overlapping")
    axa.scatter(*Pe[ok_e].T, s=14, c="tab:green", label="reachable, no overlap")
    axa.plot(*Pn[np.append(hv, hv[0])].T, "k-", lw=1.5, label="nominal range")
    rn = prob["rng_nom"]
    for ax, poses, alpha in ((axa, [tuple(prob["ref"])], 1.0),
                             (axb, [(a, c) for a in rn[0] for c in rn[1]] + [tuple(prob["ref"])], 0.6)):
        for k, (t1, t2) in enumerate(poses):
            for s, (a, e) in enumerate(pose(t1, t2)):
                ax.plot([a[0], e[0]], [a[1], e[1]], "-o", color=LINK_COLORS[s], lw=3, ms=4, alpha=alpha,
                        label=LINK_LABELS[s] if k == 0 else None)
        ax.plot(0, 0, "ks", ms=9)
        ax.plot(*O, "ks", ms=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x [m]")
    axa.set_ylabel("y [m]")
    axa.legend(fontsize=8, loc="upper left")
    axa.set_title(f"Leg {prob['name']}: f1={b['f1']:.2f}, f2={b['f2']:.2f}, extended range {100*ok_e.mean():.0f} %")
    axb.set_title("corners of the nominal range and reference pose")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def animate(res, ang, path, dt=0.01, max_frames=300, fps=20):
    """GIF of the mechanism following `ang` (N, 2): plane angles of ONE robot over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    prob, b = res["prob"], res["best"]
    ang = np.asarray(ang)[: max_frames]
    reach, free, _, bars = evaluate(prob, b["x"], b["f1"], b["f2"], (ang[:, 0], ang[:, 1]), b["branch"], detail=True)
    O = np.array([b["x"][3], -b["x"][4]])
    allp = np.vstack([np.vstack(s) for s in bars] + [np.zeros((1, 2))])
    P = bars[1][1]
    fig, ax = plt.subplots(figsize=(6, 7))

    def draw(i):
        ax.cla()
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlim(allp[:, 0].min() - 0.03, allp[:, 0].max() + 0.03)
        ax.set_ylim(allp[:, 1].min() - 0.03, allp[:, 1].max() + 0.03)
        ax.plot(*P[: i + 1].T, color="gray", lw=1, alpha=0.6)
        for s, (a, e) in enumerate(bars):
            ax.plot([a[i, 0], e[i, 0]], [a[i, 1], e[i, 1]], "-o", color=LINK_COLORS[s], lw=3, ms=5)
        ax.plot(0, 0, "ks", ms=9)
        ax.plot(*O, "ks", ms=9)
        flag = "" if (reach[i] and free[i]) else "  (unreachable or overlapping)"
        ax.set_title(f"Leg {prob['name']}: t = {i * dt:.2f} s{flag}")

    FuncAnimation(fig, draw, frames=len(ang), interval=1000 / fps).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
