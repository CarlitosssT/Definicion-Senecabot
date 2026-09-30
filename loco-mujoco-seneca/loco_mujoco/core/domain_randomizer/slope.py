from typing import Any, Sequence, Tuple, Union
from types import ModuleType

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.domain_randomizer import DomainRandomizer
from loco_mujoco.core.utils.backend import assert_backend_is_supported


@struct.dataclass
class SlopeRandomizerState:
    """
    State of the slope randomizer (one per environment).

    """

    gravity: Union[np.ndarray, jax.Array]       # gravity vector of the current episode (world = slope frame)
    gravity_dir: Union[np.ndarray, jax.Array]   # its unit direction (read by gravity_direction())
    slope_deg: Union[float, jax.Array]      # signed slope of the current episode: + uphill, - downhill
    n_steps: Union[int, jax.Array]          # steps simulated by this env since its first reset (curriculum clock)


class SlopeRandomizer(DomainRandomizer):
    """
    [SENECA LOCAL CHANGE] Walking on a uniform slope, sampled once per episode.

    The slope is simulated by tilting the gravity vector about the world y axis over the flat floor,
    which is physically the same as tilting the floor (same gravity-ground angle). The world frame is
    therefore the slope frame: root height, orientation and velocity seen by the observations, the
    reward and the terminal-state handler stay relative to the ground, so the flat-ground reference
    trajectory remains valid. A policy only perceives the slope through an observation that reads this
    state (``SlopeProjectedGravity``).

    Sign convention: the robot walks toward +x, ``slope_deg > 0`` is uphill and ``< 0`` downhill;
    gravity = |g| * (-sin(slope), 0, -cos(slope)).

    Args:
        env: The environment instance.
        slope_range_deg: [min, max] of the uniform slope distribution in degrees. ``[s, s]`` fixes the
            slope (evaluation).
        curriculum_start_deg: with ``curriculum_steps > 0``, the sampled range is clipped to [-m, m],
            where m grows linearly from ``curriculum_start_deg`` to max(|min|, |max|).
        curriculum_steps: length of that ramp in steps *per environment* (0 disables the curriculum).
            Each env counts its own steps from its first reset and does not restart the count between
            episodes, so with N parallel envs the ramp ends after N * curriculum_steps env-steps.
            Freshly reset envs (e.g. PPO validation) sample from the start range.

    """

    def __init__(self, env,
                 slope_range_deg: Sequence[float] = (-30.0, 30.0),
                 curriculum_start_deg: float = None,
                 curriculum_steps: int = 0):

        lo, hi = (float(v) for v in slope_range_deg)
        assert lo <= hi, f"slope_range_deg must be [min, max], got {list(slope_range_deg)}."
        self._lo, self._hi = lo, hi
        self._max_abs = max(abs(lo), abs(hi))
        self._curriculum = curriculum_start_deg is not None and curriculum_steps is not None \
            and curriculum_steps > 0
        if self._curriculum:
            assert lo <= 0.0 <= hi, "The curriculum clips the range toward 0, so it must contain 0."
            self._curriculum_start = float(curriculum_start_deg)
            self._curriculum_steps = float(curriculum_steps)
        self._g = float(np.linalg.norm(env.model.opt.gravity))

        super().__init__(env, slope_range_deg=[lo, hi], curriculum_start_deg=curriculum_start_deg,
                         curriculum_steps=curriculum_steps)

    def init_state(self,
                   env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType) -> SlopeRandomizerState:
        """
        Initialize the randomizer state (flat ground until the first reset samples a slope).

        Args:
            env (Any): The environment instance.
            key (Any): Random seed key.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            SlopeRandomizerState: The initialized randomizer state.

        """
        assert_backend_is_supported(backend)
        g = np.asarray(model.opt.gravity, dtype=float)        # MjModel: concrete values
        # explicit dtypes: both branches of the in-step reset (jax.lax.cond) must return identical types
        return SlopeRandomizerState(gravity=backend.asarray(g, dtype=backend.float32),
                                    gravity_dir=backend.asarray(g / np.linalg.norm(g), dtype=backend.float32),
                                    slope_deg=backend.asarray(0.0, dtype=backend.float32),
                                    n_steps=backend.asarray(0, dtype=backend.int32))

    def current_range(self, n_steps, backend: ModuleType):
        """ Slope range [min, max] in degrees available to an env that has simulated ``n_steps`` steps. """
        if not self._curriculum:
            return self._lo, self._hi
        frac = backend.minimum(n_steps / self._curriculum_steps, 1.0)
        m = self._curriculum_start + (self._max_abs - self._curriculum_start) * frac
        return backend.maximum(self._lo, -m), backend.minimum(self._hi, m)

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        Sample the slope of the new episode.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The unchanged data and the carry with the new slope.

        """
        assert_backend_is_supported(backend)
        state = carry.domain_randomizer_state

        if backend == jnp:
            key, _k = jax.random.split(carry.key)
            u = jax.random.uniform(_k)
            carry = carry.replace(key=key)
        else:
            u = np.random.uniform()

        lo, hi = self.current_range(state.n_steps, backend)
        slope_deg = backend.asarray(lo + (hi - lo) * u, dtype=backend.float32)
        th = backend.deg2rad(slope_deg)
        direction = backend.asarray(backend.stack([-backend.sin(th), backend.zeros_like(th), -backend.cos(th)]),
                                    dtype=backend.float32)

        carry = carry.replace(domain_randomizer_state=state.replace(
            gravity=self._g * direction, gravity_dir=direction, slope_deg=slope_deg))
        return data, carry

    def update(self,
               env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """
        Apply the episode's gravity to the model before the simulation step and advance the curriculum clock.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjModel, Model], Union[MjData, Data], Any]: The updated model, data, and carry.

        """
        assert_backend_is_supported(backend)
        state = carry.domain_randomizer_state

        if backend == jnp:
            model = self._set_attribute_in_model(model, "opt.gravity", state.gravity, backend)
        else:
            model.opt.gravity[:] = state.gravity

        carry = carry.replace(domain_randomizer_state=state.replace(n_steps=state.n_steps + 1))
        return model, data, carry

    def update_observation(self,
                           env: Any,
                           obs: Union[np.ndarray, jnp.ndarray],
                           model: Union[MjModel, Model],
                           data: Union[MjData, Data],
                           carry: Any,
                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """ No observation noise: the slope is perceived through ``SlopeProjectedGravity``. """
        assert_backend_is_supported(backend)
        return obs, carry

    def update_action(self,
                      env: Any,
                      action: Union[np.ndarray, jnp.ndarray],
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      carry: Any,
                      backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """ Actions are not modified. """
        assert_backend_is_supported(backend)
        return action, carry
