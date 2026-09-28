import time
import jax


def mjx_speed_test(env, n_envs):

    key = jax.random.key(0)
    keys = jax.random.split(key, n_envs + 1)
    key, env_keys = keys[0], keys[1:]

    # jit and vmap all functions needed
    rng_reset = jax.jit(jax.vmap(env.mjx_reset))
    rng_step = jax.jit(jax.vmap(env.mjx_step))
    rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))

    # reset env
    state = rng_reset(env_keys)

    step = 0
    previous_time = time.time()
    LOGGING_FREQUENCY = 100000
    while True:

        keys = jax.random.split(key, n_envs + 1)
        key, action_keys = keys[0], keys[1:]
        action = rng_sample_uni_action(action_keys)
        state = rng_step(state, action)

        step += n_envs
        if step % LOGGING_FREQUENCY == 0:
            current_time = time.time()
            print(f"{int(LOGGING_FREQUENCY / (current_time - previous_time))} steps per second.")
            previous_time = current_time
