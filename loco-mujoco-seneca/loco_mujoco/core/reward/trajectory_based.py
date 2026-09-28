from types import ModuleType
from typing import Any, Dict, Tuple, Union

import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model
from flax import struct
import numpy as np
import jax.numpy as jnp
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.observations.base import ObservationType
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid, mj_jntid2qposid, mj_jntid2qvelid
from loco_mujoco.core.utils.math import calculate_relative_site_quatities, quaternion_angular_distance
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
from loco_mujoco.core.reward.utils import out_of_bounds_action_cost


def check_traj_provided(method):
    """
    Decorator to check if trajectory handler is None. Raises ValueError if not provided.
    """
    def wrapper(self, *args, **kwargs):
        env = kwargs.get('env', None) if 'env' in kwargs else args[5]  # Assumes 'env' is the 6th positional argument
        if getattr(env, "th") is None:
            raise ValueError("TrajectoryHandler not provided, but required for trajectory-based rewards.")
        return method(self, *args, **kwargs)
    return wrapper


class TrajectoryBasedReward(Reward):

    """
    Base class for trajectory-based reward functions. These reward functions require a
    trajectory handler to compute the reward.

    """

    @property
    def requires_trajectory(self) -> bool:
        return True


class TargetVelocityTrajReward(TrajectoryBasedReward):
    """
    Reward function that computes the reward based on the deviation from the trajectory velocity. The trajectory
    velocity is provided as an observation in the environment. The reward is computed as the negative exponential
    of the squared difference between the current velocity and the goal velocity. The reward is computed for the
    x, y, and yaw velocities of the root.

    """

    def __init__(self, env: Any,
                 w_exp=10.0,
                 **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): Environment instance.
            w_exp (float, optional): Exponential weight for the reward. Defaults to 10.0.
            **kwargs (Any): Additional keyword arguments.
        """

        super().__init__(env, **kwargs)
        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]
        self._free_joint_qpos_idx = np.array(mj_jntname2qposid(self._free_jnt_name, env._model))
        self._free_joint_qvel_idx = np.array(mj_jntname2qvelid(self._free_jnt_name, env._model))
        self._w_exp = w_exp

    @check_traj_provided
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
        Computes a tracking reward based on the deviation from the trajectory velocity.
        Tracking is done on the x, y, and yaw velocities of the root.

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

        Raises:
            ValueError: If trajectory handler is not provided.

        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        def calc_local_vel(_d):
            _lin_vel_global = backend.squeeze(_d.qvel[self._free_joint_qvel_idx])[:3]
            _ang_vel_global = backend.squeeze(_d.qvel[self._free_joint_qvel_idx])[3:]
            _root_quat = R.from_quat(quat_scalarfirst2scalarlast(backend.squeeze(_d.qpos[self._free_joint_qpos_idx])[3:7]))
            _lin_vel_local = _root_quat.as_matrix().T @ _lin_vel_global
            # construct vel, x, y and yaw
            return backend.concatenate([_lin_vel_local[:2], backend.atleast_1d(_ang_vel_global[2])])

        # get root velocity from data
        vel_local = calc_local_vel(data)

        # calculate the same for the trajectory
        traj_data = env.th.traj.data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, backend)
        traj_vel_local = calc_local_vel(traj_data)

        # calculate tracking reward
        tracking_reward = backend.exp(-self._w_exp*backend.mean(backend.square(vel_local - traj_vel_local)))

        # set nan values to 0
        tracking_reward = backend.nan_to_num(tracking_reward, nan=0.0)

        return tracking_reward, carry


@struct.dataclass
class MimicRewardState:
    """
    State of MimicReward.
    """
    last_qvel: Union[np.ndarray, jnp.ndarray]
    last_action: Union[np.ndarray, jnp.ndarray]
    last_last_action: Union[np.ndarray, jnp.ndarray]  # [SENECA LOCAL CHANGE] for the action-jerk penalty


class MimicReward(TrajectoryBasedReward):
    """
    DeepMimic reward function that computes the reward based on the deviation from the trajectory. The reward is
    computed as the negative exponential of the squared difference between the current state and the trajectory state.
    The reward is computed for the joint positions, joint velocities, relative site positions,
    relative site orientations, and relative site velocities. These sites are specified in the environment properties
    and are placed at key points on the body to mimic the motion of the body.

    """

    def __init__(self, env: Any,
                 sites_for_mimic=None,
                 joints_for_mimic=None,
                 **kwargs):
        """
        Initialize the DeepMimic reward function.

        Args:
            env (Any): Environment instance.
            sites_for_mimic (List[str], optional): List of site names to mimic. Defaults to None, taking all.
            joints_for_mimic (List[str], optional): List of joint names to mimic. Defaults to None, taking all.
            **kwargs (Any): Additional keyword arguments.

        """

        super().__init__(env, **kwargs)

        # reward coefficients
        self._qpos_w_exp = kwargs.get("qpos_w_exp", 10.0)
        self._qvel_w_exp = kwargs.get("qvel_w_exp", 2.0)
        self._rpos_w_exp = kwargs.get("rpos_w_exp", 100.0)
        self._rquat_w_exp = kwargs.get("rquat_w_exp", 10.0)
        self._rvel_w_exp = kwargs.get("rvel_w_exp", 0.1)
        self._qpos_w_sum = kwargs.get("qpos_w_sum", 0.0)
        self._qvel_w_sum = kwargs.get("qvel_w_sum", 0.0)
        self._rpos_w_sum = kwargs.get("rpos_w_sum", 0.5)
        self._rquat_w_sum = kwargs.get("rquat_w_sum", 0.3)
        self._rvel_w_sum = kwargs.get("rvel_w_sum", 0.0)
        self._action_out_of_bounds_coeff = kwargs.get("action_out_of_bounds_coeff", 0.01)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 0.0)
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 0.0)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 0.0)

        # get main body name of the environment
        self.main_body_name = self._info_props["upper_body_xml_name"]
        model = env._model
        self.main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.main_body_name)
        rel_site_names = self._info_props["sites_for_mimic"] if sites_for_mimic is None else sites_for_mimic
        self._rel_site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                                       for name in rel_site_names])
        self._rel_body_ids = np.array([model.site_bodyid[site_id] for site_id in self._rel_site_ids])

        # determine qpos and qvel indices
        quat_in_qpos = []
        qpos_ind = []
        qvel_ind = []
        for i in range(model.njnt):
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joints_for_mimic is None or jnt_name in joints_for_mimic:
                qposid = mj_jntid2qposid(i, model)
                qvelid = mj_jntid2qvelid(i, model)
                qpos_ind.append(qposid)
                qvel_ind.append(qvelid)
                if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    quat_in_qpos.append(qposid[3:])
        self._qpos_ind = np.concatenate(qpos_ind)
        self._qvel_ind = np.concatenate(qvel_ind)
        quat_in_qpos = np.concatenate(quat_in_qpos)
        self._quat_in_qpos = np.array([True if q in quat_in_qpos else False for q in self._qpos_ind])

        # [SENECA LOCAL CHANGE] optionally drop the unobservable world x,y of the root from
        # the qpos position distance. The policy observes the root with FreeJointPosNoXY (no
        # world x,y), so penalizing absolute x,y is unlearnable and rewards standing still.
        # Default True = original behavior (track x,y); set track_root_xy=False to exclude it
        # (z height + orientation + joints stay tracked). See SENECA_LOCAL_CHANGES.md.
        self._track_root_xy = kwargs.get("track_root_xy", True)
        if self._track_root_xy:
            self._qpos_pos_mask = ~self._quat_in_qpos
        else:
            free_qpos = mj_jntname2qposid(self._info_props["root_free_joint_xml_name"], model)
            root_xy = np.isin(self._qpos_ind, np.array(free_qpos[:2]))
            self._qpos_pos_mask = (~self._quat_in_qpos) & (~root_xy)

        # calc mask for the root free joint velocities
        self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True

        # [SENECA LOCAL CHANGE] explicit forward-velocity tracking of the root.
        # Tracks local-frame (x, y, yaw) velocity vs. the reference. This is observable
        # by the policy (dq_root) so it is learnable, and gives a positive pull to walk
        # forward at the reference speed. See SENECA_LOCAL_CHANGES.md.
        self._free_joint_qpos_ind = np.array(mj_jntname2qposid(self._info_props["root_free_joint_xml_name"], model))
        self._rootvel_w_sum = kwargs.get("rootvel_w_sum", 0.0)
        self._rootvel_w_exp = kwargs.get("rootvel_w_exp", 1.0)

        # [SENECA LOCAL CHANGE] contact-pattern reward: fraction of feet whose ground contact
        # matches the reference's stance/swing phase. Agent contact = foot site z below
        # contact_height (sphere radius + tolerance). Reference contact = foot site z in the
        # lowest contact_ref_frac of that foot's height range in the reference (the reference is
        # kinematic and its feet don't all reach the floor). Built in init_from_traj.
        self._contact_w_sum = kwargs.get("contact_w_sum", 0.0)
        self._contact_height = kwargs.get("contact_height", 0.037)
        self._contact_ref_frac = kwargs.get("contact_ref_frac", 0.3)
        contact_sites = kwargs.get("contact_sites", None)
        if self._contact_w_sum > 0.0:
            assert contact_sites, "contact_w_sum > 0 requires contact_sites (the foot site names)."
            self._contact_site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                                               for name in contact_sites])
        self._ref_contact = None
        self._ref_split_points = None

        # [SENECA LOCAL CHANGE] smoothness penalties applied ONCE (the legacy action_rate_coeff is
        # multiplied by its coefficient twice below, so its effective weight is coeff**2).
        # action_rate_w: w * sum((a_t - a_{t-1})^2); action_jerk_w: w * sum((a_t - 2a_{t-1} + a_{t-2})^2).
        self._action_rate_w = kwargs.get("action_rate_w", 0.0)
        self._action_jerk_w = kwargs.get("action_jerk_w", 0.0)

    def init_from_traj(self, traj_handler: Any = None) -> None:
        """
        [SENECA LOCAL CHANGE] Precompute the reference per-foot contact flags for the contact-pattern reward.
        """
        if self._contact_w_sum <= 0.0 or traj_handler is None:
            return
        traj_data = traj_handler.traj.data
        foot_z = np.asarray(traj_data.site_xpos)[:, self._contact_site_ids, 2]
        lo, hi = np.percentile(foot_z, 2, axis=0), np.percentile(foot_z, 98, axis=0)
        self._ref_contact = foot_z <= lo + self._contact_ref_frac * (hi - lo)
        self._ref_split_points = np.asarray(traj_data.split_points)

    def init_state(self, env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType):
        """
        Initialize the reward state.

        Args:
            env (Any): The environment instance.
            key (Any): Key for the reward state.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            MimicRewardState: The initialized reward state.

        """
        return MimicRewardState(last_qvel=data.qvel, last_action=backend.zeros(env.info.action_space.shape[0]),
                                last_last_action=backend.zeros(env.info.action_space.shape[0]))

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType):
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
        reward_state = self.init_state(env, None, model, data, backend)
        carry = carry.replace(reward_state=reward_state)
        return data, carry

    @check_traj_provided
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
        Computes a deep mimic tracking reward based on the deviation from the trajectory. The reward is computed as the
        negative exponential of the squared difference between the current state and the trajectory state. The reward
        is computed for the joint positions, joint velocities, relative site positions, relative site orientations, and
        relative site velocities.

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

        Raises:
            ValueError: If trajectory handler is not provided.

        """
        # get current reward state
        reward_state = carry.reward_state

        # get trajectory data
        traj_data = env.th.traj.data

        # get all quantities from trajectory
        traj_data_single = traj_data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, backend)
        qpos_traj, qvel_traj = traj_data_single.qpos[self._qpos_ind], traj_data_single.qvel[self._qvel_ind]
        qpos_quat_traj = qpos_traj[self._quat_in_qpos].reshape(-1, 4)
        if len(self._rel_site_ids) > 1:
            site_rpos_traj, site_rangles_traj, site_rvel_traj =\
                calculate_relative_site_quatities(traj_data_single, self._rel_site_ids,
                                                self._rel_body_ids, model.body_rootid, backend)

        # get all quantities from the current data
        qpos, qvel = data.qpos[self._qpos_ind], data.qvel[self._qvel_ind]
        qpos_quat = qpos[self._quat_in_qpos].reshape(-1, 4)
        if len(self._rel_site_ids) > 1:
            site_rpos, site_rangles, site_rvel = (
                calculate_relative_site_quatities(data, self._rel_site_ids, self._rel_body_ids,
                                                model.body_rootid, backend))

        # calculate distances
        qpos_dist = backend.mean(backend.square(qpos[self._qpos_pos_mask] - qpos_traj[self._qpos_pos_mask]))  # [SENECA LOCAL CHANGE] mask honors track_root_xy
        qpos_dist += backend.mean(quaternion_angular_distance(qpos_quat, qpos_quat_traj, backend))
        qvel_dist = backend.mean(backend.square(qvel - qvel_traj))
        if len(self._rel_site_ids) > 1:
            rpos_dist = backend.mean(backend.square(site_rpos - site_rpos_traj))
            rangles_dist = backend.mean(backend.square(site_rangles - site_rangles_traj))
            rvel_rot_dist = backend.mean(backend.square(site_rvel[:,:3] - site_rvel_traj[:,:3]))
            rvel_lin_dist = backend.mean(backend.square(site_rvel[:,3:] - site_rvel_traj[:,3:]))

        # calculate rewards
        qpos_reward = backend.exp(-self._qpos_w_exp*qpos_dist)
        qvel_reward = backend.exp(-self._qvel_w_exp*qvel_dist)
        if len(self._rel_site_ids) > 1:
            rpos_reward = backend.exp(-self._rpos_w_exp*rpos_dist)
            rangles_reward = backend.exp(-self._rquat_w_exp*rangles_dist)
            rvel_rot_reward = backend.exp(-self._rvel_w_exp*rvel_rot_dist)
            rvel_lin_reward = backend.exp(-self._rvel_w_exp*rvel_lin_dist)

        # calculate costs
        # out of bounds action cost
        if self._action_out_of_bounds_coeff > 0.0:
            out_of_bound_reward = -out_of_bounds_action_cost(action, lower_bound=env.mdp_info.action_space.low,
                                                             upper_bound=env.mdp_info.action_space.high, backend=backend)
        else:
            out_of_bound_reward = 0.0

        # joint acceleration reward
        if self._joint_acc_coeff > 0.0:
            last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
            joint_vel = data.qvel[~self._free_joint_qvel_mask]
            acceleration_norm = backend.sum(backend.square(joint_vel - last_joint_vel) / env.dt)
            acceleration_reward = self._joint_acc_coeff * -acceleration_norm
        else:
            acceleration_reward = 0.0

        # joint torque reward
        if self._joint_torque_coeff > 0.0:
            torque_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            torque_reward = self._joint_torque_coeff * -torque_norm
        else:
            torque_reward = 0.0

        # action rate reward
        if self._action_rate_coeff > 0.0:
            action_rate_norm = backend.sum(backend.square(action - reward_state.last_action))
            action_rate_reward = self._action_rate_coeff * -action_rate_norm
        else:
            action_rate_reward = 0.0

        # total penality rewards
        total_penalities = (self._action_out_of_bounds_coeff * out_of_bound_reward
                            + self._joint_acc_coeff * acceleration_reward
                            + self._joint_torque_coeff * torque_reward
                            + self._action_rate_coeff * action_rate_reward)
        total_penalities = backend.maximum(total_penalities, -1.0)

        # [SENECA LOCAL CHANGE] root forward-velocity tracking reward (see SENECA_LOCAL_CHANGES.md)
        R = np_R if backend == np else jnp_R
        def _local_root_vel(_d):
            v = backend.squeeze(_d.qvel[self._free_joint_qvel_ind])
            quat = quat_scalarfirst2scalarlast(backend.squeeze(_d.qpos[self._free_joint_qpos_ind])[3:7])
            lin_local = R.from_quat(quat).as_matrix().T @ v[:3]
            return backend.concatenate([lin_local[:2], backend.atleast_1d(v[5])])  # x, y, yaw vel
        rootvel_dist = backend.mean(backend.square(_local_root_vel(data) - _local_root_vel(traj_data_single)))
        rootvel_reward = backend.nan_to_num(backend.exp(-self._rootvel_w_exp * rootvel_dist), nan=0.0)

        # calculate total reward
        total_reward = (self._qpos_w_sum * qpos_reward + self._qvel_w_sum * qvel_reward
                        + self._rootvel_w_sum * rootvel_reward)
        if len(self._rel_site_ids) > 1:
            total_reward = (total_reward
                        + self._rpos_w_sum * rpos_reward + self._rquat_w_sum * rangles_reward
                        + self._rvel_w_sum * rvel_rot_reward + self._rvel_w_sum * rvel_lin_reward)

        # [SENECA LOCAL CHANGE] contact-pattern reward
        if self._contact_w_sum > 0.0:
            if self._ref_contact is None:
                raise ValueError("contact_w_sum > 0 but no reference contacts: init_from_traj was not called.")
            idx = backend.asarray(self._ref_split_points)[carry.traj_state.traj_no] + carry.traj_state.subtraj_step_no
            ref_contact = backend.asarray(self._ref_contact)[idx]
            agent_contact = data.site_xpos[self._contact_site_ids, 2] <= self._contact_height
            contact_reward = backend.mean((agent_contact == ref_contact).astype(backend.float32))
            total_reward = total_reward + self._contact_w_sum * contact_reward

        # [SENECA LOCAL CHANGE] smoothness penalties (applied once, see __init__)
        if self._action_rate_w > 0.0:
            total_reward = total_reward - self._action_rate_w * backend.sum(
                backend.square(action - reward_state.last_action))
        if self._action_jerk_w > 0.0:
            total_reward = total_reward - self._action_jerk_w * backend.sum(
                backend.square(action - 2.0 * reward_state.last_action + reward_state.last_last_action))

        total_reward = total_reward + total_penalities

        # clip to positive values
        total_reward = backend.maximum(total_reward, 0.0)

        # set nan values to 0
        total_reward = backend.nan_to_num(total_reward, nan=0.0)

        # update reward state
        reward_state = reward_state.replace(last_qvel=data.qvel, last_action=action,
                                            last_last_action=reward_state.last_action)
        carry = carry.replace(reward_state=reward_state)

        return total_reward, carry
