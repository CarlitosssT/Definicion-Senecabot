"""
augment.py

Physics-preserving augmentation of a single-clip qpos/qvel trajectory into a
larger multi-cycle dataset (phase shift, L-R mirror, time scaling, smooth noise).

Pulled out of the bottom of Ovine_data_visualizer.ipynb. Two corrections vs the
notebook:
  * resampling goes through the shared ``data_loader.resample_to_n``,
  * the smooth-noise finite difference uses dt = 1/frequency (the real per-frame
    spacing of a normalized cycle), not the previous 1/CYCLE_FRAMES which was a
    2x velocity error.

Joint axis conventions (confirmed from senecabot_loco.xml): FR/FL share identical
joint definitions, as do BR/BL, so an L-R mirror is a pure value swap with no
sign negation on the joint angles.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from simulation.config import pipeline as P
from simulation.core.data_loader import resample_to_n

# fr<->fl, br<->bl  (joint block indices 0..11)
_MIRROR_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


# ---------------------------------------------------------------------------
# Gait-cycle detection and cycle extraction
# ---------------------------------------------------------------------------
def detect_cycles(qpos, prominence=0.04, distance=25):
    """Cycle boundaries from FR-hip-angle troughs (qpos[:, 7])."""
    fr_hip = qpos[:, 7]
    troughs, _ = find_peaks(-fr_hip, prominence=prominence, distance=distance)
    return troughs


def extract_and_normalize(qpos, qvel, boundaries,
                          target=None, min_frames=30, max_frames=200):
    """
    Cut the trajectory at ``boundaries`` and resample each segment to ``target``
    frames. Root XY is zeroed per cycle so cycles can be stacked without
    positional drift; root Z and orientation are kept (they carry gait info).
    """
    target = target or P.CYCLE_FRAMES
    cycles_q, cycles_v = [], []
    for i in range(len(boundaries) - 1):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        if not (min_frames <= e - s <= max_frames):
            continue

        r_q = resample_to_n(qpos[s:e], target)
        r_v = resample_to_n(qvel[s:e], target)

        r_q[:, 0] -= r_q[0, 0]   # zero root X
        r_q[:, 1] -= r_q[0, 1]   # zero root Y

        cycles_q.append(r_q)
        cycles_v.append(r_v)
    return cycles_q, cycles_v


# ---------------------------------------------------------------------------
# Augmentation operators
# ---------------------------------------------------------------------------
def aug_phase_shift(q, v, frac):
    """Roll the cycle to a different stride phase; bridge the root-XY seam."""
    n = len(q)
    k = int(n * frac) % n
    D_x, D_y = float(q[-1, 0]), float(q[-1, 1])

    q2 = np.roll(q, k, axis=0).copy()
    v2 = np.roll(v, k, axis=0).copy()
    q2[k:, 0] += D_x
    q2[k:, 1] += D_y
    q2[:, 0] -= q2[0, 0]
    q2[:, 1] -= q2[0, 1]
    return q2, v2


def aug_mirror_lr(q, v):
    """Left-right reflection about the sagittal (XZ) plane."""
    q2, v2 = q.copy(), v.copy()
    q2[:, 1] = -q[:, 1]          # root Y
    v2[:, 1] = -v[:, 1]
    q2[:, 4] = -q[:, 4]          # quat qx
    q2[:, 6] = -q[:, 6]          # quat qz
    v2[:, 3] = -v[:, 3]          # roll rate
    v2[:, 5] = -v[:, 5]          # yaw rate
    q2[:, 7:19] = q[:, 7:19][:, _MIRROR_PERM]
    v2[:, 6:18] = v[:, 6:18][:, _MIRROR_PERM]
    return q2, v2


def aug_time_scale(q, v, scale, target=None):
    """Speed variation: scale<1 faster, scale>1 slower. qvel divided by scale."""
    target = target or P.CYCLE_FRAMES
    n = len(q)
    mid = max(20, int(n * scale))
    q2 = resample_to_n(resample_to_n(q, mid), target)
    v2 = resample_to_n(resample_to_n(v, mid), target) / scale
    return q2, v2


def aug_smooth_noise(q, v, frequency, noise_std=0.008, sigma=4.0, rng=None):
    """
    Smooth noise on joint angles only (qpos 7..18); qvel of perturbed joints
    recomputed by finite difference at the true per-frame spacing dt=1/frequency.
    """
    if rng is None:
        rng = np.random.default_rng()
    q2, v2 = q.copy(), v.copy()
    for j in range(7, 19):
        q2[:, j] += gaussian_filter1d(rng.standard_normal(len(q)) * noise_std, sigma=sigma)
    dt = 1.0 / frequency
    for j in range(6, 18):       # qvel[6..17] <- qpos[7..18]
        v2[:, j] = np.gradient(q2[:, j + 1], dt)
    return q2, v2


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate_augmented(qpos, qvel, frequency, seed=42):
    """
    Build the full augmented dataset from a single clip.

    Returns dict with 'qpos', 'qvel', 'time', 'split_points', 'frequency', 'meta'.
    """
    troughs = detect_cycles(qpos)
    cycles_q, cycles_v = extract_and_normalize(qpos, qvel, troughs)
    if not cycles_q:
        raise ValueError("No valid gait cycles detected for augmentation.")

    q_list, v_list, meta = [], [], []

    def register(q, v, tag, base):
        q_list.append(q)
        v_list.append(v)
        meta.append({"cycle": base, "type": tag})

    for idx, (cq, cv) in enumerate(zip(cycles_q, cycles_v)):
        register(cq, cv, "original", idx)

        mq, mv = aug_mirror_lr(cq, cv)
        register(mq, mv, "mirror", idx)

        for frac, tag in [(0.25, "phase_25"), (0.50, "phase_50"), (0.75, "phase_75")]:
            pq, pv = aug_phase_shift(cq, cv, frac)
            register(pq, pv, tag, idx)
            pmq, pmv = aug_mirror_lr(pq, pv)
            register(pmq, pmv, tag + "_mirror", idx)

        for scale, tag in [(0.82, "fast_18pct"), (0.91, "fast_9pct"),
                           (1.10, "slow_10pct"), (1.22, "slow_22pct")]:
            sq, sv = aug_time_scale(cq, cv, scale)
            register(sq, sv, tag, idx)

        for s in range(4):
            nq, nv = aug_smooth_noise(cq, cv, frequency,
                                      rng=np.random.default_rng(s * 7 + idx))
            register(nq, nv, f"noise_{s}", idx)

        for s in range(2):
            mnq, mnv = aug_smooth_noise(mq, mv, frequency,
                                        rng=np.random.default_rng(s * 13 + 100 + idx))
            register(mnq, mnv, f"mirror_noise_{s}", idx)

    C = P.CYCLE_FRAMES
    all_qpos = np.concatenate(q_list, axis=0)
    all_qvel = np.concatenate(v_list, axis=0)
    # Each cycle restarts its own clock at the true spacing 1/frequency.
    all_time = np.tile(np.arange(C) / frequency, len(q_list))
    split_points = np.arange(0, len(q_list) + 1, dtype=np.int32) * C

    return {
        "qpos": all_qpos,
        "qvel": all_qvel,
        "time": all_time,
        "split_points": split_points,
        "frequency": float(frequency),
        "meta": meta,
    }


def save_augmented(aug, path=None):
    """Save an augmented dataset dict to .npz (default PROC_DIR / AUGMENTED_NAME)."""
    path = path or (P.PROC_DIR / P.AUGMENTED_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qpos=aug["qpos"],
        qvel=aug["qvel"],
        time=aug["time"],
        split_points=aug["split_points"],
        frequency=aug["frequency"],
    )
    return path
