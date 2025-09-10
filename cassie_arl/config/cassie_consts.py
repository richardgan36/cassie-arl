from jax import numpy as jnp

### Indices
BASE_IDX = jnp.array([0, 1, 2, 3, 4, 5, 6])
MOTOR_IDX = jnp.array([7, 8, 9, 14, 20, 21, 22, 23, 28, 34])
MOTOR_VEL_IDX = jnp.array([6, 7, 8, 12, 18, 19, 20, 21, 25, 31])

LEFT_HIP_ROLL_IDX = 0
LEFT_HIP_YAW_IDX = 1
RIGHT_HIP_ROLL_IDX = 5
RIGHT_HIP_YAW_IDX = 6

STANDING_POSE = jnp.array([
    0.0,        # Base X
    0.0,        # Base Y
    0.95,       # Base Z
    0.0,        # Base roll
    0.0,        # Base pitch
    0.0,        # Base yaw
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

BASE_STANDING_POSE = STANDING_POSE[:6]
MOTORS_STANDING_POSE = STANDING_POSE[6:]

STANDING_PELVIS_RPY = STANDING_POSE[3:6]

FALLING_THRESHOLD = 0.55