from typing import Dict, Any, Union, Tuple
from types import ModuleType

import numpy as np
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.terminal_state_handler.base import TerminalStateHandler
from loco_mujoco.core.utils import mj_jntname2qposid
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast, gravity_direction
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class UprightTerminalStateHandler(TerminalStateHandler):
    """
    [SENECA LOCAL CHANGE] Fall detection that does not depend on a reference trajectory: the state is
    terminal when the trunk's up axis is tilted more than ``max_tilt_deg`` from gravity-up, or when the
    root is lower than ``min_height`` above the ground (world z; the slope normal with SlopeRandomizer).

    The gravity direction is the episode's (``SlopeRandomizer`` state, else ``model.opt.gravity``), so on a
    slope the trunk may stay level, parallel to the ground or anything in between. The reference-based
    ``RootPoseTrajTerminalStateHandler`` instead ends the episode when the trunk leaves the reference's
    orientation range (+30 deg), i.e. whenever it stops being parallel to the slope.

    Args:
        env: The environment instance.
        max_tilt_deg: Maximum angle between the trunk's up axis and gravity-up.
        min_height: Minimum root height above the ground [m] (SenecaBot: trunk box half-height 0.06 m,
            the v140 gait keeps the root between 0.31 and 0.40 m).

    """

    def __init__(self, env: Any, max_tilt_deg: float = 50.0, min_height: float = 0.15):
        super().__init__(env, max_tilt_deg=max_tilt_deg, min_height=min_height)
        self._cos_max_tilt = float(np.cos(np.deg2rad(max_tilt_deg)))
        self._min_height = float(min_height)
        self._root_qpos_ind = np.array(mj_jntname2qposid(self._info_props["root_free_joint_xml_name"], env._model))

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """ Stateless: nothing to reset. """
        assert_backend_is_supported(backend)
        return data, carry

    def is_absorbing(self, env: Any, obs: np.ndarray, info: Dict[str, Any], data: MjData,
                     carry: Any) -> Union[bool, Any]:
        """ Check if the current state is terminal. Function for CPU Mujoco. """
        return self._is_absorbing_compat(env, obs, info, data, carry, backend=np)

    def mjx_is_absorbing(self, env: Any, obs: jnp.ndarray, info: Dict[str, Any], data: Data,
                         carry: Any) -> Union[bool, Any]:
        """ Check if the current state is terminal. Function for Mjx. """
        return self._is_absorbing_compat(env, obs, info, data, carry, backend=jnp)

    def _is_absorbing_compat(self, env: Any, obs, info: Dict[str, Any], data, carry: Any,
                             backend: ModuleType) -> Union[bool, Any]:
        """ Check if the current state is terminal. Compatible with both CPU Mujoco and Mjx. """
        R = np_R if backend == np else jnp_R
        root = data.qpos[self._root_qpos_ind]
        up_body = R.from_quat(quat_scalarfirst2scalarlast(root[3:7])).apply(backend.array([0.0, 0.0, 1.0]))

        up_world = -gravity_direction(carry, env.model, backend)

        tilt_cond = backend.less(backend.dot(up_body, up_world), self._cos_max_tilt)
        height_cond = backend.less(root[2], self._min_height)
        return backend.logical_or(tilt_cond, height_cond), carry
