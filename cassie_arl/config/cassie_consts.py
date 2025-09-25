import jax
from jax import numpy as jnp
from dataclasses import dataclass


FALLING_THRESHOLD = 0.55
TARSUS_HIT_GROUND_THRESHOLD = 0.1  # Height (m) below which tarsus is considered to have hit ground
FOOT_OFFSET = 0.057  # Height (m) of foot from ground when standing


@dataclass(frozen=True)
class StandingPose:
    PELVIS_RPY: jax.Array = jnp.array([0.0, 0.0, 0.0])
    MOTOR_ANGLES: jax.Array = jnp.array([
        0.0,        # Left hip roll
        0.0,        # Left hip yaw
        0.4544,     # Left hip pitch
        -1.21,      # Left knee
        -1.643,     # Left foot
        0.0,        # Right hip roll
        0.0,        # Right hip yaw
        0.4544,     # Right hip pitch
        -1.21,      # Right knee
        -1.643,     # Right foot
    ])
    MOTOR_TORQUES: jax.Array = jnp.array([  # TODO: replace with actual values
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])


@dataclass(frozen=True)
class JntRangeIdx:
    MOTORS: jax.Array = jnp.array([
        1, 2, 3, 5, 11,
        12, 13, 14, 16, 22
    ])


@dataclass(frozen=True)
class QPosIdx:
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
    BASE: jax.Array = jnp.array([0, 1, 2, 3, 4, 5])
    BASE_LIN_VEL: jax.Array = jnp.array([0, 1, 2])
    MOTORS: jax.Array = jnp.array([
        6, 7, 8, 12, 18,
        19, 20, 21, 25, 31
    ])
