"""
cyclic.py

Build a seamless *looping* reference from a single mocap walk capture.

Why this exists
---------------
loco-mujoco's ``GoalTrajMimic`` never terminates the episode at the end of the
clip (it inherits ``Goal.mjx_is_done -> False``), and the training ``horizon``
(10 s) is far longer than a single ~2 s capture. So the trajectory handler WRAPS
the clip, and the goal observation goes NaN at that wrap. This was verified to be
intrinsic and policy-independent: it fires even for a zero-torque policy and even
for a perfectly clean cyclic clip. The episode then ends via the env's
``done |= isnan(obs)`` guard -- a benign truncation, but it caps every episode at
one clip length and prevents learning a sustained gait. (See CLAUDE.md, the
2026-06-06 root-cause correction.)

The fix, without editing loco-mujoco: tile the gait into ONE long contiguous
sub-trajectory. Internal cycle-joins are then normal consecutive frames (no wrap,
no NaN); only the single final boundary wraps. With the tiled clip much longer
than the horizon, almost every episode runs full-length without ever reaching it,
while random-start (RSI) is preserved.

Two steps:
  1. ``extract_cycle`` -- find the best phase-matched single steady-state gait
     cycle in the raw capture (autocorrelation period + minimal-seam window
     search). The raw capture is rest->walk->rest, so its own endpoints don't
     match (a ~73 deg seam); a mid-capture window does (~5 deg).
  2. ``tile_cycle`` -- repeat that cycle, accumulating the per-cycle root-XY
     translation so the base keeps walking forward instead of teleporting. The
     observation is world-XY invariant, so the accumulation is for physical /
     playback sanity, not correctness.
"""

import numpy as np

from simulation.config import pipeline as P

# qpos layout: [root_xyz(3), root_quat(4), 12 hinge joints]. Joint angles live at 7:19.
_JOINT_SLICE = slice(7, 19)
# qvel layout: [root_lin(3), root_ang(3), 12 hinge joint velocities]. Legs live at 6:18.
_JOINT_SLICE_V = slice(6, 18)

# The 12 hinges, in qpos[7:19] / qvel[6:18] order, are:
#   [fr_hip, fr_knee, fr_ankle,  fl_hip, fl_knee, fl_ankle,
#    br_hip, br_knee, br_ankle,  bl_hip, bl_knee, bl_ankle]
# Bilateral (left<->right) swap: fr<->fl and br<->bl. In senecabot_loco.xml the
# left and right legs share the SAME joint axis sign and the SAME joint range
# (e.g. fr/fl hip both "25 105", br/bl hip both "-105 -40"), so the contralateral
# swap is a pure column permutation -- NO sign flip.
_LR_SWAP = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])


def _autocorr_period(signal, lo, hi):
    """Dominant period (in frames) of a 1-D signal, searched in [lo, hi]."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / max(ac[0], 1e-12)
    band = range(max(lo, 1), min(hi, len(ac) - 1))
    peaks = [l for l in band if ac[l] > ac[l - 1] and ac[l] >= ac[l + 1]]
    if peaks:
        return max(peaks, key=lambda l: ac[l])
    return int((lo + hi) // 2)


def _seam_cost(qpos, qvel, s, period):
    """How discontinuous the loop is if we cut a cycle [s, s+period): the wrap
    joins frame s+period-1 back to frame s, so compare frame s+period to frame s
    (one-period apart, same gait phase). Lower is smoother."""
    j_jump = np.abs(np.rad2deg(qpos[s + period, _JOINT_SLICE] - qpos[s, _JOINT_SLICE])).max()
    v_jump = np.abs(qvel[s + period, 6:] - qvel[s, 6:]).max()
    z_jump = abs(qpos[s + period, 2] - qpos[s, 2])
    return j_jump + 10.0 * v_jump + 100.0 * z_jump, j_jump


def _phase_shift_periodic(block, shift):
    """Circularly shift a (P, k) *periodic* signal forward by ``shift`` frames.
    ``shift`` may be fractional (the gait period is often odd, so half a period
    is not an integer) -- linear interpolation with wrap-around handles it."""
    block = np.asarray(block, dtype=np.float64)
    P = block.shape[0]
    t = (np.arange(P) + shift) % P
    i0 = np.floor(t).astype(int) % P
    i1 = (i0 + 1) % P
    frac = (t - np.floor(t))[:, None]
    return (1.0 - frac) * block[i0] + frac * block[i1]


def symmetrize_cycle(cyc_q, cyc_v, mode="average"):
    """Enforce left/right (bilateral) symmetry on ONE periodic gait cycle.

    A symmetric quadruped gait (walk/trot/pace) obeys ``left(t) = right(t + T/2)``.
    The raw ovine-derived reference violates this with a per-joint postural bias
    (the front-left leg sits ~5-9 deg more flexed than front-right), which shows
    up visually as the fl_knee/fl_foot "trip". This re-imposes the symmetry on the
    12 hinge joints only (the root pose is left untouched):

        ``average``      legs_sym(t) = 0.5*( legs(t) + swap(legs(t + P/2)) )
        ``mirror_right`` legs_sym(t) =            swap(legs_R(t + P/2))   onto L,
                         right side kept as-is, left replaced by shifted right.
        ``mirror_left``  the mirror image of ``mirror_right``.

    qpos and qvel receive the IDENTICAL linear operator, so they stay mutually
    consistent. The cycle is periodic, so the result is also exactly periodic
    (the loop seam in the leg joints becomes ~0 by construction).

    Args:
        cyc_q (np.ndarray): (P, 19) one-period qpos.
        cyc_v (np.ndarray): (P, 18) one-period qvel.
        mode (str): ``average`` (default), ``mirror_right`` or ``mirror_left``.

    Returns:
        (cyc_q_sym, cyc_v_sym, residual_deg): symmetrized copies plus the
        pre-fix peak left/right joint-bias in degrees (for logging).
    """
    cyc_q = np.asarray(cyc_q, dtype=np.float64).copy()
    cyc_v = np.asarray(cyc_v, dtype=np.float64).copy()

    def _sym(block):
        shifted_swapped = _phase_shift_periodic(block, block.shape[0] / 2.0)[:, _LR_SWAP]
        if mode == "average":
            return 0.5 * (block + shifted_swapped)
        if mode == "mirror_right":
            # keep the right legs (fr,br = cols 0,1,2 / 6,7,8); set the left legs
            # (fl,bl) to their phase-shifted right counterparts.
            out = block.copy()
            out[:, [3, 4, 5, 9, 10, 11]] = shifted_swapped[:, [3, 4, 5, 9, 10, 11]]
            return out
        if mode == "mirror_left":
            out = block.copy()
            out[:, [0, 1, 2, 6, 7, 8]] = shifted_swapped[:, [0, 1, 2, 6, 7, 8]]
            return out
        raise ValueError(f"unknown symmetrize mode: {mode!r}")

    legs_q = cyc_q[:, _JOINT_SLICE]
    # peak per-joint left/right bias BEFORE the fix (mean over the cycle of the
    # contralateral, half-period-aligned difference) -- for the log line.
    aligned = _phase_shift_periodic(legs_q, legs_q.shape[0] / 2.0)[:, _LR_SWAP]
    residual_deg = float(np.rad2deg(np.abs(legs_q.mean(0) - aligned.mean(0)).max()))

    cyc_q[:, _JOINT_SLICE] = _sym(legs_q)
    cyc_v[:, _JOINT_SLICE_V] = _sym(cyc_v[:, _JOINT_SLICE_V])
    return cyc_q, cyc_v, residual_deg


def extract_cycle(qpos, qvel, period_range=(110, 150), search_pad=80):
    """Locate the best phase-matched single steady-state gait cycle in the capture.

    Args:
        qpos (np.ndarray): (T, 19) capture positions.
        qvel (np.ndarray): (T, 18) capture velocities.
        period_range (tuple): (lo, hi) gait period search bounds, in frames.
        search_pad (int): keep the window away from the rest->walk / walk->rest
            transients at the clip ends.

    Returns:
        dict: ``start``, ``period``, ``seam_deg`` (residual joint discontinuity),
        and ``dxy`` (per-cycle root-XY translation, to accumulate when tiling).
    """
    qpos = np.asarray(qpos); qvel = np.asarray(qvel)
    n = qpos.shape[0]
    lo, hi = period_range
    period = _autocorr_period(qpos[:, 7], lo, hi)         # fr_hip drives the period

    best = None
    for pd in range(lo, hi):
        for s in range(search_pad, n - pd - 10):
            cost, j_jump = _seam_cost(qpos, qvel, s, pd)
            if best is None or cost < best[0]:
                best = (cost, s, pd, j_jump)
    _, s, pd, j_jump = best
    dxy = (qpos[s + pd, :2] - qpos[s, :2]).copy()         # forward translation / cycle
    return dict(start=int(s), period=int(pd), seam_deg=float(j_jump), dxy=dxy)


def tile_cycle_arrays(cyc_q, cyc_v, dxy, n_tiles):
    """Repeat a single (already-extracted) cycle ``n_tiles`` times into one
    contiguous trajectory, advancing the root XY by ``dxy`` each repeat so the
    base keeps walking forward.

    Returns (qpos_loop (n_tiles*P, 19), qvel_loop (n_tiles*P, 18))."""
    cyc_q = np.asarray(cyc_q); cyc_v = np.asarray(cyc_v)
    qpos_tiles, qvel_tiles = [], []
    for k in range(n_tiles):
        blk = cyc_q.copy()
        blk[:, 0] += k * dxy[0]
        blk[:, 1] += k * dxy[1]
        qpos_tiles.append(blk)
        qvel_tiles.append(cyc_v.copy())                   # velocity is phase-periodic
    return np.concatenate(qpos_tiles, axis=0), np.concatenate(qvel_tiles, axis=0)


def tile_cycle(qpos, qvel, start, period, dxy, n_tiles):
    """Slice the cycle [start, start+period) out of the full capture and tile it.
    Thin wrapper over :func:`tile_cycle_arrays` (kept for backward compatibility)."""
    qpos = np.asarray(qpos); qvel = np.asarray(qvel)
    return tile_cycle_arrays(qpos[start:start + period].copy(),
                             qvel[start:start + period].copy(), dxy, n_tiles)


# Foot sites used for the no-slip root speed (same sites as the contact-pattern reward).
_FOOT_SITES = ("fr_foot_mimic", "fl_foot_mimic", "br_foot_mimic", "bl_foot_mimic")


def no_slip_root_speed(model, cyc_q, frequency, stance_frac=0.3, foot_sites=_FOOT_SITES):
    """Forward root speed (m/s) that the cycle's leg motion supports without foot slip.

    The mocap root translation is the sheep's T13 marker, i.e. the SHEEP's walking speed.
    The robot's legs are shorter, so with the same joint angles its feet sweep far less
    per cycle and a reference moving at the sheep's speed has every foot sliding.

    With the root held still, each foot's forward velocity relative to the body is
    computed by forward kinematics. A foot is in stance when it is in the lowest
    ``stance_frac`` of its height range over the cycle (the same criterion as the
    contact-pattern reward, ``contact_ref_frac``). The returned speed ``v`` minimizes the
    squared foot slip ``sum_stance (v_foot_rel + v)^2`` over all feet and stance frames,
    i.e. ``v = -mean(v_foot_rel over stance frames)``.

    Returns:
        (v, info) with per-foot mean stance velocity and x sweep for logging.
    """
    import mujoco
    data = mujoco.MjData(model)
    sids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in foot_sites]
    P_ = cyc_q.shape[0]
    pos = np.zeros((P_, len(sids), 3))
    for i in range(P_):
        data.qpos[:] = cyc_q[i]
        data.qpos[0:3] = 0.0                              # body still: foot motion relative to it
        mujoco.mj_kinematics(model, data)
        pos[i] = data.site_xpos[sids]
    z, x = pos[..., 2], pos[..., 0]
    lo, hi = z.min(0), z.max(0)
    stance = z <= lo + stance_frac * (hi - lo)
    # periodic central difference (the cycle wraps onto itself)
    vx = (np.roll(x, -1, 0) - np.roll(x, 1, 0)) * frequency / 2.0
    v = float(-vx[stance].mean())
    info = dict(stance_vx=[float(vx[stance[:, k], k].mean()) for k in range(len(sids))],
                sweep_x=[float(x[:, k].max() - x[:, k].min()) for k in range(len(sids))],
                stance_frac=[float(stance[:, k].mean()) for k in range(len(sids))])
    return v, info


def scale_root_speed(cyc_q, cyc_v, dxy, scale):
    """Scale the cycle's root XY travel (positions relative to the cycle start, the linear
    XY velocity and the per-cycle ``dxy``) by ``scale``. Height, orientation and joints are
    untouched; the within-cycle speed fluctuation keeps its shape."""
    cyc_q = np.asarray(cyc_q, dtype=np.float64).copy()
    cyc_v = np.asarray(cyc_v, dtype=np.float64).copy()
    cyc_q[:, 0:2] = cyc_q[0, 0:2] + scale * (cyc_q[:, 0:2] - cyc_q[0, 0:2])
    cyc_v[:, 0:2] *= scale
    return cyc_q, cyc_v, np.asarray(dxy, dtype=np.float64) * scale


def make_looping_reference(qpos, qvel, frequency, loop_duration_s=60.0,
                           period_range=(110, 150), search_pad=80,
                           symmetrize=True, symmetrize_mode="average",
                           root_speed="mocap", model=None, stance_frac=0.3):
    """End-to-end: extract a clean cycle and tile it to ~``loop_duration_s`` of
    continuous walking, as ONE contiguous sub-trajectory.

    Args:
        qpos, qvel (np.ndarray): raw capture arrays.
        frequency (float): capture rate (Hz).
        loop_duration_s (float): target length of the looped reference. Make this
            comfortably larger than ``horizon`` (in seconds) so few episodes ever
            reach the single final wrap boundary.
        root_speed: forward speed of the root. ``"mocap"`` keeps the sheep's T13 travel;
            ``"no_slip"`` rescales it to :func:`no_slip_root_speed` (needs ``model``); a
            number sets that speed in m/s.
        model (mujoco.MjModel): robot model, required for ``root_speed="no_slip"``.
        stance_frac (float): stance criterion for ``"no_slip"`` (lowest fraction of foot height).

    Returns:
        dict with ``qpos`` (M,19), ``qvel`` (M,18), ``split_points`` ([0, M]),
        and ``info`` (the extraction summary + n_tiles), for logging.
    """
    info = extract_cycle(qpos, qvel, period_range, search_pad)
    period = info["period"]
    cycle_s = period / float(frequency)
    n_tiles = max(1, int(np.ceil(loop_duration_s / cycle_s)))

    # Slice the single steady-state cycle, then (optionally) enforce left/right
    # symmetry on it BEFORE tiling. The cycle is periodic, so the half-period
    # alignment is well-defined and the tiled clip stays seamlessly loopable.
    cyc_q = qpos[info["start"]:info["start"] + period].copy()
    cyc_v = qvel[info["start"]:info["start"] + period].copy()
    info["symmetrized"] = bool(symmetrize)
    if symmetrize:
        cyc_q, cyc_v, residual_deg = symmetrize_cycle(cyc_q, cyc_v, mode=symmetrize_mode)
        info["symmetrize_mode"] = symmetrize_mode
        info["lr_bias_deg_prefix"] = residual_deg   # peak L/R joint bias removed
    # Root forward speed: the mocap value is the sheep's, which the robot's legs can't support.
    info["root_speed_mocap"] = float(np.linalg.norm(info["dxy"]) / cycle_s)
    if root_speed == "no_slip":
        if model is None:
            raise ValueError('root_speed="no_slip" needs the robot model (model=...).')
        target, info["no_slip"] = no_slip_root_speed(model, cyc_q, frequency, stance_frac)
    elif root_speed == "mocap":
        target = info["root_speed_mocap"]
    else:
        target = float(root_speed)
    scale = target / info["root_speed_mocap"]
    cyc_q, cyc_v, info["dxy"] = scale_root_speed(cyc_q, cyc_v, info["dxy"], scale)
    info["root_speed"] = target
    info["root_speed_scale"] = scale
    qpos_loop, qvel_loop = tile_cycle_arrays(cyc_q, cyc_v, info["dxy"], n_tiles)
    info["n_tiles"] = n_tiles
    info["cycle_s"] = cycle_s
    info["frames"] = qpos_loop.shape[0]
    return dict(
        qpos=qpos_loop,
        qvel=qvel_loop,
        split_points=[0, qpos_loop.shape[0]],
        info=info,
    )
