import jax
from jax import numpy as jnp
from dataclasses import dataclass


FALLING_THRESHOLD = 0.55  # Cassie is considered to have fallen if pelvis falls below this height (m)
TARSUS_HIT_GROUND_THRESHOLD = 0.1  # Height (m) below which tarsus is considered to have hit ground
FOOT_CONTACT_THRESHOLD = 0.015  # Height (m) below which foot is considered to be in contact with ground
FOOT_OFFSET = 0.057  # Height (m) of foot from ground when standing


@dataclass(frozen=True)
class JntRangeIdx:
    """The indices of the motor joints in MuJoCo's joint range array."""
    MOTORS: jax.Array = jnp.array([
        1, 2, 3, 5, 11,
        12, 13, 14, 16, 22
    ])


@dataclass(frozen=True)
class QPosIdx:
    """The indices of various components in the qpos array."""
    BASE: jax.Array = jnp.array([0, 1, 2, 3, 4, 5, 6])
    BASE_XY: jax.Array = jnp.array([0, 1])
    BASE_HEIGHT: jax.Array = jnp.array([2])
    BASE_QUAT: jax.Array = jnp.array([3, 4, 5, 6])
    MOTORS: jax.Array = jnp.array([
        7, 8, 9, 14, 20,
        21, 22, 23, 28, 34
    ])


@dataclass(frozen=True)
class QVelIdx:
    """The indices of various components in the qvel array."""
    BASE: jax.Array = jnp.array([0, 1, 2, 3, 4, 5])
    BASE_LIN_VEL: jax.Array = jnp.array([0, 1, 2])
    BASE_ANG_VEL: jax.Array = jnp.array([3, 4, 5])
    MOTORS: jax.Array = jnp.array([
        6, 7, 8, 12, 18,
        19, 20, 21, 25, 31
    ])
