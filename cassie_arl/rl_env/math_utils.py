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


def quat_mul(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Quaternion multiplication q = q1 * q2.
    Rotates by q2 first, then q1.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_conjugate(q: jax.Array) -> jax.Array:
    """Quaternion conjugate (w, -x, -y, -z)."""
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def quat_apply(q: jax.Array, v: jax.Array) -> jax.Array:
    """Rotate vector v by quaternion q (q * v * q_conj)."""
    # Promote v to pure quaternion
    vq = jnp.concatenate([jnp.array([0.0]), v])
    return quat_mul(quat_mul(q, vq), quat_conjugate(q))[1:]


def vec_world_to_body(base_quat: jax.Array, v_world: jax.Array) -> jax.Array:
    """Rotate world-frame vector into body frame."""
    # base_quat maps body->world, so body_vec = q_conj * v_world * q
    q_conj = quat_conjugate(base_quat)
    return quat_apply(q_conj, v_world)


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


def quat2mat(quat: jax.Array) -> jax.Array:
    """
    Convert quaternion [w, x, y, z] to a 3x3 rotation matrix R that maps
    vectors from the base frame to the world frame.

    Args:
        quat: jax.Array of shape (4,) or (..., 4)

    Returns:
        jax.Array of shape (3, 3) or (..., 3, 3)
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    R = jnp.stack([
        jnp.stack([1-2*(y**2+z**2), 2*(x*y - w*z), 2*(x*z + w*y)], axis=-1),
        jnp.stack([2*(x*y + w*z), 1-2*(x**2+z**2), 2*(y*z - w*x)], axis=-1),
        jnp.stack([2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x**2+y**2)], axis=-1),
    ], axis=-2)
    return R


def gravity_in_base_frame(base_quat: jax.Array) -> jax.Array:
    """Compute gravity vector in the base frame given the base quaternion."""
    # quat: [w, x, y, z]
    R = quat2mat(base_quat)
    g_world = jnp.array([0., 0., -9.81])
    # R maps base->world, so R.T maps world->base
    return R.T @ g_world


def vec_xy_world_to_base(
        vec_world: jax.Array,
        base_quat: jax.Array,
    ) -> jax.Array:
    """Convert a 2D vector in the XY plane from world frame to base frame."""
    base_euler = quat2euler(base_quat)
    yaw = base_euler[..., 2]

    c = jnp.cos(-yaw)
    s = jnp.sin(-yaw)

    x = vec_world[..., 0]
    y = vec_world[..., 1]
    # Rotation by -yaw to go from world -> pelvis
    return jnp.array([c * x - s * y, s * x + c * y])
