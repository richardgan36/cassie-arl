from __future__ import annotations
from flax import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class BiquadFilter:
    """Second-order (biquad) low-pass Butterworth filter (vectorized).

    Coefficients are scalars (broadcast) and state arrays are shape (n_channels,).
    Difference equation (a0=1):
        y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]
    Immutable + JAX-friendly.
    """
    b0: jax.Array
    b1: jax.Array
    b2: jax.Array
    a1: jax.Array
    a2: jax.Array
    x1: jax.Array
    x2: jax.Array
    y1: jax.Array
    y2: jax.Array

    @classmethod
    def create(cls, coeffs, n_channels: int) -> "BiquadFilter":
        b0, b1, b2, a1, a2 = coeffs
        z = jnp.zeros((n_channels,))
        return cls(
            b0=jnp.asarray(b0), b1=jnp.asarray(b1), b2=jnp.asarray(b2),
            a1=jnp.asarray(a1), a2=jnp.asarray(a2),
            x1=z, x2=z, y1=z, y2=z
        )

    def apply(self, x: jax.Array) -> tuple[jax.Array, "BiquadFilter"]:
        y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        new_self = self.replace(x1=x, x2=self.x1, y1=y, y2=self.y1)
        return y, new_self


def design_butterworth_biquad(fc_hz: float, fs_hz: float):
    """Design 2nd-order Butterworth low-pass biquad.

    Returns tuple (b0,b1,b2,a1,a2) for use with difference equation (a0=1).
    Uses bilinear transform; fc_hz < fs_hz/2.
    """
    fc = jnp.array(fc_hz)
    fs = jnp.array(fs_hz)
    omega_c = jnp.tan(jnp.pi * fc / fs)  # pre-warp
    omega_c2 = omega_c * omega_c
    sqrt2 = jnp.sqrt(2.0)
    denom = 1.0 + sqrt2 * omega_c + omega_c2
    b0 = omega_c2 / denom
    b1 = 2.0 * b0
    b2 = b0
    a1 = (2.0 * (omega_c2 - 1.0)) / denom
    a2 = (1.0 - sqrt2 * omega_c + omega_c2) / denom
    return b0, b1, b2, a1, a2
