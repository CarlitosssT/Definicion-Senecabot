from test_conf import *

# set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
# jax.config.update('jax_exec_time_optimization_effort', 1.0)
# jax.config.update('jax_memory_fitting_effort', 1.0)
print(f"Jax backend device: {jax.default_backend()} \n")

_TOLERANCE = 1e-5


def test_trajectory_generator(standing_trajectory, falling_trajectory):

    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions_jax = generate_test_trajectories(expert_traj, nominal_traj, "jax", horizon=100,
                                                 reward_type="NoReward")
    transitions_np = generate_test_trajectories(expert_traj, nominal_traj, "numpy", horizon=100,
                                                reward_type="NoReward")

    assert jnp.allclose(transitions_np.observations, transitions_jax.observations, atol=_TOLERANCE, rtol=_TOLERANCE).all()
    assert jnp.allclose(transitions_np.actions, transitions_jax.actions, atol=_TOLERANCE, rtol=_TOLERANCE).all()
    assert jnp.allclose(transitions_np.rewards, transitions_jax.rewards, atol=_TOLERANCE, rtol=_TOLERANCE).all()
    assert jnp.allclose(transitions_np.next_observations, transitions_jax.next_observations, atol=_TOLERANCE, rtol=_TOLERANCE).all()
