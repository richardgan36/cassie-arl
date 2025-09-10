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