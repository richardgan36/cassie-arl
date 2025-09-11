import jax
from jax import numpy as jnp


def quat2euler(q: jax.Array) -> jax.Array:
    """
    Convert a quaternion [w, x, y, z] to Euler angles (XYZ order, radians) using JAX.
    
    Args:
        q: jax.Array of shape (..., 4), quaternion [w, x, y, z]
    
    Returns:
        jax.Array of shape (..., 3), Euler angles [roll, pitch, yaw] in radians
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x**2 + y**2)
    roll = jnp.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = jnp.clip(t2, -1.0, 1.0)  # avoid NaNs due to numerical errors
    pitch = jnp.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y**2 + z**2)
    yaw = jnp.arctan2(t3, t4)
    
    return jnp.stack([roll, pitch, yaw], axis=-1)


def euler2quat(euler: jax.Array) -> jax.Array:
    """
    Convert Euler angles [roll, pitch, yaw] (XYZ order, radians) to quaternion [w, x, y, z] using JAX.

    Args:
        euler: jax.Array of shape (..., 3), Euler angles [roll, pitch, yaw] in radians

    Returns:
        jax.Array of shape (..., 4), quaternion [w, x, y, z]
    """
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]

    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return jnp.stack([w, x, y, z], axis=-1)


def angle_diff(angle1: jax.Array, angle2: jax.Array) -> jax.Array:
    """
    Compute the wrapped difference between two angles in radians.
    Wraps the result to [-pi, pi].

    Args:
        angle1: jax.Array, the minuend angle
        reference: jax.Array, the subtrahend angle

    Returns:
        Wrapped difference (angle - reference) in [-pi, pi]
    """
    diff = angle1 - angle2
    return (diff + jnp.pi) % (2 * jnp.pi) - jnp.pi