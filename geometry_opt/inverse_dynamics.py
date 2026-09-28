"""
Inverse dynamics of the recorded gait on a candidate design.

For every recorded state (q, v, a) of the gait:
  1. mj_inverse with all constraints disabled gives the generalized force the design needs,
       qfrc_inv = M(q) a + c(q, v) - qfrc_passive      (joint damping included, armature in M)
  2. Known generalized forces are kept as recorded in the simulated gait: joint-limit forces (the
     hind ankles/hips lean on their mechanical stops), joint dry friction, leg-leg self-contacts and
     the torsional/rolling contact torques of the condim-6 feet. The unactuated floating base
     (first 6 rows) must then be balanced by the ground forces:
       qfrc_inv[:6] - known[:6] = sum_k J_k[:, :6]^T f_k
     The contact forces f_k at the recorded contact points are re-solved for the new masses:
       min  W^2 ||A f - b||^2 + lambda ||f - f_recorded||^2   s.t. f in the friction pyramid
     (NNLS on the pyramid edges). With the trained masses f_recorded already satisfies it, so the
     solution reproduces the simulation; with heavier designs the forces grow as little as needed.
  3. Joint (motor) torques:  tau = qfrc_inv[joint] - known[joint] - sum_k J_k[:, joint]^T f_k
     (if LIMITS_CARRY_LOAD is False the motor also supplies the joint-limit torque)
  4. Link internal loads: the contact forces are applied as xfrc_applied and mj_rnePostConstraint
     gives cfrc_int, the wrench each body receives from its parent. It is moved to the joint point
     (proximal end) and to the child's joint (distal end) and split in the link frame (link axis =
     body z): axial force N, shear V, bending moment Mb, torsion T.
"""
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import nnls

from . import config as C
from .model_builder import JOINT_NAMES, LINK_NAMES

# square friction pyramid |fx|, |fy| <= mu fz (circumscribes the elliptic cone the simulator uses)
_EDGE_T = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)


@dataclass
class Loads:
    tau: np.ndarray          # (S, 12) motor torque [N·m], actuator order (JOINT_NAMES)
    tau_damp: np.ndarray     # (S, 12) part of tau spent against the model's joint damping [N·m]
    tau_limit: np.ndarray    # (S, 12) torque carried by the joint-limit stops [N·m]
    omega: np.ndarray        # (S, 12) joint speed [rad/s]
    N: np.ndarray            # (S, 12, 2) axial force [N]   at (proximal, distal) end, LINK_NAMES order
    V: np.ndarray            # (S, 12, 2) shear force [N]
    Mb: np.ndarray           # (S, 12, 2) bending moment [N·m]
    T: np.ndarray            # (S, 12, 2) torsion [N·m]
    F: np.ndarray            # (S, 12) resultant force at the proximal joint [N]
    f_contact: np.ndarray    # (S, K, 3) ground forces [N]
    root_residual: np.ndarray  # (S, 6) unbalanced base wrench after the contact solve
    hinge_check: float       # max |moment about hinge from cfrc_int - joint generalized force| [N·m]

    def summary(self):
        a = np.abs
        return dict(
            tau_max=a(self.tau).max(0), tau_rms=np.sqrt((self.tau ** 2).mean(0)),
            tau_damp_at_max=a(self.tau_damp[a(self.tau).argmax(0), np.arange(self.tau.shape[1])]),
            tau_damp_max=a(self.tau_damp).max(0),
            omega_max_rpm=a(self.omega).max(0) * 60 / (2 * np.pi),
            N_max=a(self.N).max(axis=(0, 2)), F_max=self.F.max(0),
            Mb_max=self.Mb.max(axis=(0, 2)), T_max=a(self.T).max(axis=(0, 2)), V_max=self.V.max(axis=(0, 2)),
        )


def _split(Fw, Mw, z):
    """World force/moment -> axial, shear, torsion, bending w.r.t. the unit axis z."""
    n = Fw @ z
    t = Mw @ z
    return n, np.linalg.norm(Fw - n * z), t, np.linalg.norm(Mw - t * z)


def run(model: mujoco.MjModel, gait: dict) -> Loads:
    m = model
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    d = mujoco.MjData(m)
    S = gait["qpos"].shape[0]
    names = [str(n) for n in gait["body_names"]]
    body_map = {i: m.body(n).id for i, n in enumerate(names)}          # gait body idx -> this model
    act_dof = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
    assert tuple(m.actuator(i).name for i in range(m.nu)) == JOINT_NAMES
    links = [m.body(n).id for n in LINK_NAMES]
    child = {b: next((c for c in range(m.nbody) if m.body_parentid[c] == b and c != b), None) for b in links}
    jnt_of_link = [m.body_jntadr[b] for b in links]
    root_id = m.body_rootid

    tau = np.zeros((S, m.nu))
    tau_damp = np.zeros((S, m.nu))
    tau_limit = np.zeros((S, m.nu))
    jacr = np.zeros((3, m.nv))
    Nn, Vv, Mb, Tt = (np.zeros((S, len(links), 2)) for _ in range(4))
    Fres = np.zeros((S, len(links)))
    K = gait["c_body"].shape[1]
    f_out = np.zeros((S, K, 3))
    resid = np.zeros((S, 6))
    hinge_err = 0.0
    jacp = np.zeros((3, m.nv))

    for s in range(S):
        d.qpos[:], d.qvel[:], d.qacc[:] = gait["qpos"][s], gait["qvel"][s], gait["qacc"][s]
        d.xfrc_applied[:] = 0.0
        mujoco.mj_inverse(m, d)
        qinv = d.qfrc_inverse.copy()
        # known generalized forces (kept as simulated)
        known = gait["q_dryfric"][s] + gait["q_limit"][s]
        ks = np.where(gait["c_body"][s] >= 0)[0]
        for k in ks:                                            # condim-6 contact torques
            b = body_map[int(gait["c_body"][s, k])]
            mujoco.mj_jac(m, d, None, jacr, d.xpos[b], b)
            known = known + jacr.T @ gait["c_torque"][s, k]
        self_c = np.where(gait["sc_body"][s, :, 0] >= 0)[0]
        for k in self_c:                                        # leg-leg self-contacts
            b0, b1 = (body_map[int(x)] for x in gait["sc_body"][s, k])
            p, fk = gait["sc_pos"][s, k], gait["sc_force"][s, k]
            mujoco.mj_jac(m, d, jacp, None, p, b1)
            known = known + jacp.T @ fk
            mujoco.mj_jac(m, d, jacp, None, p, b0)
            known = known - jacp.T @ fk

        # ---- ground forces for this design --------------------------------------------------
        Js, E_blocks = [], []
        for k in ks:
            b = body_map[int(gait["c_body"][s, k])]
            mujoco.mj_jac(m, d, jacp, None, gait["c_pos"][s, k], b)
            Js.append(jacp.copy())
            mu = gait["c_mu"][s, k]
            E_blocks.append(np.column_stack([[mu * tx, mu * ty, 1.0] for tx, ty in _EDGE_T]))  # 3x4
        if len(ks):
            A = np.hstack([J[:, :6].T for J in Js])                                       # 6 x 3n
            E = np.zeros((3 * len(ks), 4 * len(ks)))
            for i, Eb in enumerate(E_blocks):
                E[3 * i:3 * i + 3, 4 * i:4 * i + 4] = Eb
            f_ref = gait["c_force"][s, ks].ravel()
            lam = np.sqrt(C.CONTACT_REG_LAMBDA)
            M_ls = np.vstack([C.ROOT_EQ_WEIGHT * A @ E, lam * E])
            y_ls = np.concatenate([C.ROOT_EQ_WEIGHT * (qinv[:6] - known[:6]), lam * f_ref])
            beta, _ = nnls(M_ls, y_ls, maxiter=2000)
            f = (E @ beta).reshape(-1, 3)
            gen_contact = sum(J.T @ fk for J, fk in zip(Js, f))
            f_out[s, ks] = f
        else:
            f = np.zeros((0, 3))
            gen_contact = np.zeros(m.nv)
        resid[s] = qinv[:6] - known[:6] - gen_contact[:6]
        tau[s] = qinv[act_dof] - known[act_dof] - gen_contact[act_dof]
        if not C.LIMITS_CARRY_LOAD:
            tau[s] += gait["q_limit"][s, act_dof]
        tau_damp[s] = -d.qfrc_passive[act_dof]
        tau_limit[s] = gait["q_limit"][s, act_dof]

        # ---- internal link wrenches -----------------------------------------------------------
        for k, fk in zip(ks, f):
            b = body_map[int(gait["c_body"][s, k])]
            d.xfrc_applied[b, :3] += fk
            d.xfrc_applied[b, 3:] += np.cross(gait["c_pos"][s, k] - d.xipos[b], fk) + gait["c_torque"][s, k]
        for k in self_c:
            b0, b1 = (body_map[int(x)] for x in gait["sc_body"][s, k])
            p, fk = gait["sc_pos"][s, k], gait["sc_force"][s, k]
            d.xfrc_applied[b1, :3] += fk
            d.xfrc_applied[b1, 3:] += np.cross(p - d.xipos[b1], fk)
            d.xfrc_applied[b0, :3] -= fk
            d.xfrc_applied[b0, 3:] -= np.cross(p - d.xipos[b0], fk)
        mujoco.mj_rnePostConstraint(m, d)
        for li, b in enumerate(links):
            z = d.xmat[b].reshape(3, 3)[:, 2]
            c = d.subtree_com[root_id[b]]
            F = d.cfrc_int[b, 3:]
            Mp = d.cfrc_int[b, :3] + np.cross(c - d.xpos[b], F)            # moment at the joint point
            Nn[s, li, 0], Vv[s, li, 0], Tt[s, li, 0], Mb[s, li, 0] = _split(F, Mp, z)
            Fres[s, li] = np.linalg.norm(F)
            ax = d.xaxis[jnt_of_link[li]]
            # sanity check: moment about the hinge = motor + passive + limit + dry friction
            # - armature*qacc (the rotor inertia acts on the dof, not through the link wrench)
            j = act_dof[li]
            expect = (qinv[j] - known[j] - gen_contact[j] + d.qfrc_passive[j] + gait["q_limit"][s, j]
                      + gait["q_dryfric"][s, j] - m.dof_armature[j] * d.qacc[j])
            hinge_err = max(hinge_err, abs(Mp @ ax - expect))
            ch = child[b]
            if ch is not None:                                                # distal end = child's joint
                Fc = -d.cfrc_int[ch, 3:]
                Mc = -(d.cfrc_int[ch, :3] + np.cross(c - d.xpos[ch], d.cfrc_int[ch, 3:]))
                Nn[s, li, 1], Vv[s, li, 1], Tt[s, li, 1], Mb[s, li, 1] = _split(Fc, Mc, z)

    omega = gait["qvel"][:, act_dof]
    return Loads(tau=tau, tau_damp=tau_damp, tau_limit=tau_limit, omega=omega, N=Nn, V=Vv, Mb=Mb, T=Tt, F=Fres, f_contact=f_out,
                 root_residual=resid, hinge_check=hinge_err)


def validate(loads: Loads, gait: dict, total_mass: float):
    """Iteration-0 check: inverse dynamics on the trained model vs the forward simulation."""
    err = loads.tau - gait["tau_sim"]
    ref = np.abs(gait["tau_sim"]).max(0)
    w = total_mass * 9.81
    return dict(
        tau_rms_err=np.sqrt((err ** 2).mean(0)),
        tau_peak_ratio=np.abs(loads.tau).max(0) / np.maximum(ref, 1e-9),
        tau_corr=np.array([np.corrcoef(loads.tau[:, i], gait["tau_sim"][:, i])[0, 1] for i in range(err.shape[1])]),
        root_force_residual_max_pct=100 * np.linalg.norm(loads.root_residual[:, :3], axis=1).max() / w,
        root_force_residual_mean_pct=100 * np.linalg.norm(loads.root_residual[:, :3], axis=1).mean() / w,
        contact_force_diff_max=np.abs(loads.f_contact - gait["c_force"]).max(),
        hinge_check_Nm=loads.hinge_check,
    )
