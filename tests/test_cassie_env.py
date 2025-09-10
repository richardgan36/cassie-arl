import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# Prepend repo root so tests can import the package
repo_root = Path(__file__).resolve().parent.parent
print(repo_root)
sys.path.insert(0, str(repo_root))

from cassie_arl.rl_env.my_cassie_env import CassieEnv


@pytest.fixture
def env():
    return CassieEnv()


def test_action_size_matches_model(env):
    assert env.action_size == env.mjx_model.nu


def test_reset_returns_valid_state(env):
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    # obs is a 1D jax array
    assert isinstance(state.obs, jax.Array)
    assert state.obs.ndim == 1

    # data qpos / qvel sizes match model
    assert state.data.qpos.shape[0] == env.mjx_model.nq
    assert state.data.qvel.shape[0] == env.mjx_model.nv

    # reward / done are scalars
    assert jnp.asarray(state.reward).shape == ()
    assert jnp.asarray(state.done).shape == ()

    # info contains rng and step
    assert "rng" in state.info
    assert "step" in state.info
    assert int(state.info["step"]) == 0


def test_step_progresses_and_shapes(env):
    key = jax.random.PRNGKey(1)
    state = env.reset(key)

    zero_action = jnp.zeros(env.action_size, dtype=jnp.float32)
    next_state = env.step(state, zero_action)

    # observation shape preserved
    assert next_state.obs.shape == state.obs.shape

    # reward is scalar and finite
    r = float(next_state.reward)
    assert np.isfinite(r)

    # done is scalar numeric (0/1)
    assert next_state.done.shape == ()
    assert np.isscalar(float(next_state.done))

    # info.step incremented
    assert int(next_state.info.get("step", -1)) == int(state.info.get("step", 0)) + 1


def test_action_scaling_maps_bounds(env):
    nu = env.action_size
    ones = jnp.ones((nu,), dtype=jnp.float32)
    neg_ones = -ones

    upper = env._actuator_soft_bounds[1]
    lower = env._actuator_soft_bounds[0]

    scaled_upper = env._action_norm2actual(ones)
    scaled_lower = env._action_norm2actual(neg_ones)

    assert jnp.allclose(scaled_upper, upper, atol=1e-6)
    assert jnp.allclose(scaled_lower, lower, atol=1e-6)


def test_multiple_steps_no_exception_and_obs_consistent(env):
    key = jax.random.PRNGKey(2)
    state = env.reset(key)

    rng = key
    for i in range(5):
        rng, sub = jax.random.split(rng)
        action = jax.random.uniform(sub, shape=(env.action_size,), minval=-1.0, maxval=1.0)
        state = env.step(state, action)

        # obs shape consistent each step
        assert isinstance(state.obs, jax.Array)
        assert state.obs.ndim == 1


def test_get_termination_returns_bool_array(env):
    key = jax.random.PRNGKey(3)
    state = env.reset(key)
    term = env._get_termination(state.data)

    assert isinstance(term, jax.Array)
    assert term.shape == ()
    assert term.dtype == jnp.bool_
