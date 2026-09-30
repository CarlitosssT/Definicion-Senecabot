from types import ModuleType
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R
import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast, gravity_direction


@struct.dataclass
class SlopeLocomotionRewardState:
    """
    State of SlopeLocomotionReward.
    """
    last_action: Union[np.ndarray, jax.Array]
    last_last_action: Union[np.ndarray, jax.Array]
    air_time: Union[np.ndarray, jax.Array]      # per foot, time since lift-off [s]
    in_contact: Union[np.ndarray, jax.Array]    # per foot, hysteresis contact state


class SlopeLocomotionReward(Reward):
    """
    [SENECA LOCAL CHANGE] Task reward for walking on slopes, with no imitation term: nothing is compared
    with a reference trajectory, so the policy is free to change posture, gait and speed with the slope.

    The world frame is the slope frame (``SlopeRandomizer`` tilts gravity, not the floor): x is the fall
    line (uphill for slope > 0), z the slope normal, and heights are measured from the ground. Terms, per
    control step:

        + vel_w * exp(-((v_x - target_speed)^2 + v_y^2) / vel_sigma^2)   progress along the fall line
        + alive_w                                                        per step without falling
        - yaw_rate_w * w_z^2               turning about the slope normal
        - vz_w * v_z^2                     trunk bouncing (velocity along the slope normal)
        - ang_vel_w * (w_x^2 + w_y^2)      trunk roll/pitch rates (body frame)
        - roll_w * g_y^2                   lateral tilt w.r.t. gravity (g: unit gravity in the trunk frame);
                                           the trunk pitch is free
        - power_w * sum|tau * qdot|        mechanical power of the joints
        - torque_w * sum tau^2
        - action_rate_w * sum (a_t - a_{t-1})^2  and  action_jerk_w * sum (a_t - 2a_{t-1} + a_{t-2})^2
                                           (the smoothness penalties of MimicReward, unchanged)
        - slip_w * sum over touching feet of |v_foot,xy|^2       feet sliding on the ground
        + air_time_w * sum over touchdowns of (t_air - air_time_min)
                                           steps shorter than air_time_min are penalized (micro-hops)

    A foot touches the ground when its site is below ``contact_height`` (sphere radius + 2 mm); for the
    air time it stays in contact until it rises above ``release_height`` (hysteresis against flicker).
    The total is clipped at 0 like MimicReward: a negative per-step reward would make ending the episode
    attractive. Default weights were calibrated on the flat-ground gait of the v140 agent (~1.6 per step,
    penalties ~20% of it).

    """

    def __init__(self, env: Any,
                 target_speed: float = 1.4,
                 vel_sigma: float = 1.0,
                 vel_w: float = 1.5,
                 alive_w: float = 0.5,
                 yaw_rate_w: float = 0.5,
                 vz_w: float = 0.5,
                 ang_vel_w: float = 0.05,
                 roll_w: float = 5.0,
                 power_w: float = 5e-4,
                 torque_w: float = 1e-4,
                 action_rate_w: float = 0.5,
                 action_jerk_w: float = 0.5,
                 slip_w: float = 0.2,
                 air_time_w: float = 3.0,
                 air_time_min: float = 0.2,
                 contact_height: float = 0.027,
                 release_height: float = 0.037,
                 foot_sites: Sequence[str] = ("fr_foot", "fl_foot", "br_foot", "bl_foot")):

        # no **kwargs: a misspelled weight in the config raises instead of being silently ignored
        super().__init__(env)

        self._target_speed, self._vel_sigma2 = float(target_speed), float(vel_sigma) ** 2
        self._w = dict(vel=vel_w, alive=alive_w, yaw_rate=yaw_rate_w, vz=vz_w, ang_vel=ang_vel_w, roll=roll_w,
                       power=power_w, torque=torque_w, action_rate=action_rate_w, action_jerk=action_jerk_w,
                       slip=slip_w, air_time=air_time_w)
        self._air_time_min = float(air_time_min)
        self._contact_height, self._release_height = float(contact_height), float(release_height)

        model = env._model
        root = self._info_props["root_free_joint_xml_name"]
        self._root_qpos_ind = np.array(mj_jntname2qposid(root, model))
        self._root_qvel_ind = np.array(mj_jntname2qvelid(root, model))
        self._joint_qvel_mask = np.ones(model.nv, dtype=bool)
        self._joint_qvel_mask[self._root_qvel_ind] = False
        self._foot_site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in foot_sites])
        assert np.all(self._foot_site_ids >= 0), f"Unknown foot site in {list(foot_sites)}."
        self._foot_body_ids = np.array(model.site_bodyid[self._foot_site_ids])
        self._foot_root_ids = np.array(model.body_rootid[self._foot_body_ids])

    def init_state(self, env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType) -> SlopeLocomotionRewardState:
        """
        Initialize the reward state.

        Args:
            env (Any): The environment instance.
            key (Any): Key for the reward state.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            SlopeLocomotionRewardState: The initialized reward state.

        """
        n_act, n_feet = env.info.action_space.shape[0], len(self._foot_site_ids)
        # feet start "in contact" with an air time of air_time_min: a foot already in the air at the reset
        # lands with a neutral air-time reward instead of a penalty for the part of its swing it missed
        return SlopeLocomotionRewardState(last_action=backend.zeros(n_act), last_last_action=backend.zeros(n_act),
                                          air_time=backend.full(n_feet, self._air_time_min),
                                          in_contact=backend.ones(n_feet, dtype=bool))

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the reward state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated data and carry.

        """
        carry = carry.replace(reward_state=self.init_state(env, None, model, data, backend))
        return data, carry

    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
        """
        Computes the task reward (see class docstring).

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.

        """
        R = np_R if backend == np else jnp_R
        w = self._w
        rs = carry.reward_state

        # ---- trunk kinematics (world = slope frame) --------------------------------------------------
        quat = data.qpos[self._root_qpos_ind][3:7]
        rot = R.from_quat(quat_scalarfirst2scalarlast(quat))
        vel = data.qvel[self._root_qvel_ind]
        lin, ang_body = vel[:3], vel[3:]                 # free joint: linear world, angular body frame
        yaw_rate = rot.apply(ang_body)[2]

        g_body = rot.inv().apply(gravity_direction(carry, model, backend))

        progress = backend.exp(-(backend.square(lin[0] - self._target_speed) + backend.square(lin[1]))
                               / self._vel_sigma2)
        stability = (w["yaw_rate"] * backend.square(yaw_rate) + w["vz"] * backend.square(lin[2])
                     + w["ang_vel"] * backend.sum(backend.square(ang_body[:2]))
                     + w["roll"] * backend.square(g_body[1]))

        # ---- effort and smoothness -----------------------------------------------------------------------
        tau = data.qfrc_actuator[self._joint_qvel_mask]
        qdot = data.qvel[self._joint_qvel_mask]
        effort = (w["power"] * backend.sum(backend.abs(tau * qdot)) + w["torque"] * backend.sum(backend.square(tau))
                  + w["action_rate"] * backend.sum(backend.square(action - rs.last_action))
                  + w["action_jerk"] * backend.sum(backend.square(action - 2.0 * rs.last_action
                                                                  + rs.last_last_action)))

        # ---- feet: slip while touching, air time at touchdown --------------------------------------
        foot_pos = data.site_xpos[self._foot_site_ids]
        cvel = data.cvel[self._foot_body_ids]                                 # [angular, linear] at subtree com
        offset = foot_pos - data.subtree_com[self._foot_root_ids]
        foot_vel = cvel[:, 3:] + backend.cross(cvel[:, :3], offset)
        touching = foot_pos[:, 2] <= self._contact_height
        slip = backend.sum(backend.where(touching, backend.sum(backend.square(foot_vel[:, :2]), axis=1), 0.0))

        contact = backend.where(rs.in_contact, foot_pos[:, 2] <= self._release_height, touching)
        first_contact = backend.logical_and(contact, backend.logical_not(rs.in_contact))
        air_time = rs.air_time + env.dt
        air_reward = backend.sum(backend.where(first_contact, air_time - self._air_time_min, 0.0))
        air_time = backend.where(contact, 0.0, air_time)

        total = (w["vel"] * progress + w["alive"] - stability - effort - w["slip"] * slip
                 + w["air_time"] * air_reward)
        total = backend.nan_to_num(backend.maximum(total, 0.0), nan=0.0)

        carry = carry.replace(reward_state=rs.replace(last_action=action, last_last_action=rs.last_action,
                                                      air_time=air_time, in_contact=contact))
        return total, carry
