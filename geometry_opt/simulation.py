"""
Simulation: run a trained policy and record the gait it actually achieves.

1. MJX/MJWarp rollout of the deterministic policy on the ORIGINAL thesis model (the one it was
   trained on), GAIT_N_ENVS parallel robots x GAIT_STEPS control steps. Robots whose episode ends
   inside the window (trajectory wrap) are discarded.
2. CPU MuJoCo replay of every recorded state with the recorded actuator torques (mj_forward):
   gives the instantaneous generalized accelerations qacc, the ground contacts (body, point, force,
   contact torque of the condim-6 feet, friction), and the generalized forces of the other
   constraints: joint limits (the hind ankles/hips lean on their stops), joint dry friction and
   leg-leg self-contacts.

The recorded kinematics (qpos, qvel, qacc) are the design requirement: every candidate design must
perform THIS gait. Its loads are then obtained by inverse dynamics (inverse_dynamics.py).
The result is cached in results/gait_<key>.npz. Arrays are flattened step-major: sample index =
step * n_envs + robot (reshape to (n_per_env, n_envs, ...) to follow one robot in time).
"""
import os

import mujoco
import numpy as np

from . import config as C

MAX_CONTACTS = 12
MAX_SELF = 6
_CT = mujoco.mjtConstraint


def _make_env(agent_conf, reference, spec=None):
    from omegaconf import OmegaConf
    from loco_mujoco import TaskFactory
    from loco_mujoco.trajectory.dataclasses import Trajectory
    from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf

    cfg = OmegaConf.to_container(agent_conf.config.experiment.env_params, resolve=True)
    cfg["headless"] = True
    if spec is not None:
        cfg["spec"] = spec
    return TaskFactory.get_factory_cls("ImitationFactory").make(
        **cfg, custom_dataset_conf=CustomDatasetConf(traj=Trajectory.load(str(reference))))


def record_gait(key, use_cache=True):
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = C.RESULTS_DIR / f"gait_{key}.npz"
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from loco_mujoco.algorithms import PPOJax
    from simulation.analysis.debug_agent import rollout

    spec = C.MODELS[key]
    agent_conf, agent_state = PPOJax.load_agent(str(spec["agent"]))
    env = _make_env(agent_conf, spec["reference"])
    m = env.get_model()
    dt = float(env.dt)

    d = rollout(env, agent_conf, agent_state, C.GAIT_N_ENVS, C.GAIT_STEPS,
                deterministic=True, seed=C.GAIT_SEED, collect_dynamics=True)
    done = d["done"].astype(bool)
    envs = np.where(~done.any(0))[0]
    sl = slice(C.GAIT_WARMUP, C.GAIT_STEPS)
    qpos = d["qpos"][sl][:, envs].reshape(-1, m.nq)
    qvel = d["qvel"][sl][:, envs].reshape(-1, m.nv)
    act_dof = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
    tau_sim = d["qfrc_actuator"][sl][:, envs][..., act_dof].reshape(-1, m.nu)
    n_steps = C.GAIT_STEPS - C.GAIT_WARMUP
    vx = (d["qpos"][C.GAIT_STEPS - 1, envs, 0] - d["qpos"][C.GAIT_WARMUP, envs, 0]) / ((n_steps - 1) * dt)

    # ---- CPU replay: instantaneous accelerations and ground contacts --------------------------
    body_names = [m.body(b).name for b in range(m.nbody)]
    floor = m.geom("floor").id
    S = qpos.shape[0]
    qacc = np.zeros((S, m.nv))
    c_body = -np.ones((S, MAX_CONTACTS), dtype=np.int32)
    c_pos = np.zeros((S, MAX_CONTACTS, 3))
    c_force = np.zeros((S, MAX_CONTACTS, 3))
    c_mu = np.zeros((S, MAX_CONTACTS))
    c_torque = np.zeros((S, MAX_CONTACTS, 3))
    q_limit = np.zeros((S, m.nv))
    q_dryfric = np.zeros((S, m.nv))
    sc_body = -np.ones((S, MAX_SELF, 2), dtype=np.int32)       # self-contacts: force acts +f on [1], -f on [0]
    sc_pos = np.zeros((S, MAX_SELF, 3))
    sc_force = np.zeros((S, MAX_SELF, 3))
    res = np.zeros(m.nv)
    dd = mujoco.MjData(m)
    f6 = np.zeros(6)
    overflow = 0
    for s in range(S):
        dd.qpos[:], dd.qvel[:], dd.ctrl[:] = qpos[s], qvel[s], tau_sim[s]
        mujoco.mj_forward(m, dd)
        qacc[s] = dd.qacc
        # generalized forces of joint limits and joint dry friction (efc rows of those types)
        ne = dd.nefc
        types = dd.efc_type[:ne]
        for mask_type, dst in ((_CT.mjCNSTR_LIMIT_JOINT, q_limit), (_CT.mjCNSTR_FRICTION_DOF, q_dryfric)):
            vec = np.zeros_like(dd.efc_force)
            vec[:ne] = np.where(types == mask_type, dd.efc_force[:ne], 0.0)
            mujoco.mj_mulJacTVec(m, dd, res, vec)
            dst[s] = res
        k = ks = 0
        for i in range(dd.ncon):
            con = dd.contact[i]
            if con.efc_address < 0:
                continue
            mujoco.mj_contactForce(m, dd, i, f6)
            R = con.frame.reshape(3, 3).T                    # contact frame -> world
            fw, tw = R @ f6[:3], R @ f6[3:]                  # force/torque exerted by geom1 on geom2
            if floor in (con.geom1, con.geom2):
                other = con.geom2 if con.geom1 == floor else con.geom1
                if k >= MAX_CONTACTS:
                    overflow += 1
                    continue
                if con.geom1 != floor:                        # make it the force ON the robot
                    fw, tw = -fw, -tw
                c_body[s, k] = m.geom_bodyid[other]
                c_pos[s, k] = con.pos
                c_force[s, k] = fw
                c_torque[s, k] = tw
                c_mu[s, k] = con.friction[0]
                k += 1
            elif ks < MAX_SELF:                               # leg-leg self-contact (kept as recorded)
                sc_body[s, ks] = (m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2])
                sc_pos[s, ks] = con.pos
                sc_force[s, ks] = fw
                ks += 1
            else:
                overflow += 1
    if overflow:
        print(f"[simulation] warning: {overflow} samples had more than {MAX_CONTACTS} floor contacts")

    out = dict(qpos=qpos, qvel=qvel, qacc=qacc, tau_sim=tau_sim, c_body=c_body, c_pos=c_pos,
               c_force=c_force, c_torque=c_torque, c_mu=c_mu, q_limit=q_limit, q_dryfric=q_dryfric,
               sc_body=sc_body, sc_pos=sc_pos, sc_force=sc_force,
               body_names=np.array(body_names), dt=dt,
               n_envs=len(envs), n_per_env=n_steps, speed=float(vx.mean()),
               actuator_names=np.array([m.actuator(i).name for i in range(m.nu)]),
               mass_trained=float(m.body_subtreemass[m.body("base").id]))
    np.savez_compressed(cache, **out)
    return out
