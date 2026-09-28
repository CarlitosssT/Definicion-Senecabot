import numpy
from math import isclose

from test_conf import *


# set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")


def _create_env(backend, control_type="DefaultControl", control_params=None):
    DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}

    mjx_env = DummyHumamoidEnv(enable_mjx=backend == "jax",
                               control_type=control_type,
                               control_params=control_params,
                               **DEFAULTS)

    return mjx_env


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_DefaultControl(backend, mock_random):
    mjx_env = _create_env(backend, "DefaultControl")

    seed = 0
    key = jax.random.PRNGKey(seed)
    
    action = np.random.randn(mjx_env.info.action_space.shape[0])

    if backend == 'numpy':
        mjx_env.reset(key)
        mjx_env.step(action)

        print(f'\nmjx_env._data.ctrl[0]: {mjx_env._data.ctrl[0]}\nmjx_env._data.ctrl[6]: {mjx_env._data.ctrl[6]}')
        assert isclose(mjx_env._data.ctrl[0], 0.08000000000000002, abs_tol=1e-6)
        assert isclose(mjx_env._data.ctrl[6], 0.08000000000000002, abs_tol=1e-6)
    else:
        state = mjx_env.mjx_reset(key)
        action = jnp.array(action)
        state = mjx_env.mjx_step(state, action)

        print(f'\nstate.data.ctrl[0]: {state.data.ctrl[0]}\nstate.data.ctrl[6]: {state.data.ctrl[6]}')
        assert isclose(state.data.ctrl[0].item(), 0.08000000566244125, abs_tol=1e-6)
        assert isclose(state.data.ctrl[6].item(), 0.08000000566244125, abs_tol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_PDControl(backend, mock_random):
    p_gain = 1.0
    d_gain = 0.1
    control_params=dict(p_gain=p_gain, d_gain=d_gain)
    mjx_env = _create_env(backend, "PDControl", control_params=control_params)

    seed = 0
    key = jax.random.PRNGKey(seed)

    action = np.random.randn(mjx_env.info.action_space.shape[0])

    if backend == 'numpy':
        mjx_env.reset(key)
        mjx_env.step(action)
        print(f'\nmjx_env._data.ctrl[0]: {mjx_env._data.ctrl[0]}\nmjx_env._data.ctrl[6]: {mjx_env._data.ctrl[6]}')

        assert isclose(mjx_env._data.ctrl[0], -0.4, abs_tol=1e-6)
        assert isclose(mjx_env._data.ctrl[6], -0.4, abs_tol=1e-6)
    else:
        state = mjx_env.mjx_reset(key)
        action = jnp.array(action)
        state = mjx_env.mjx_step(state, action)

        print(f'\nstate.data.ctrl[0]: {state.data.ctrl[0]}\nstate.data.ctrl[6]: {state.data.ctrl[6]}')
        assert isclose(state.data.ctrl[0].item(), -0.4000000059604645, abs_tol=1e-6)
        assert isclose(state.data.ctrl[6].item(), -0.4000000059604645, abs_tol=1e-6)

