"""
Push system for applying external disturbances to Cassie during training.

This module provides a JAX-friendly push system that can apply random forces
and torques to specified bodies during simulation to improve policy robustness.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from jax import lax
import mujoco.mjx as mjx
from flax import struct
from ml_collections import config_dict

import cassie_arl.rl_env.math_utils as math_utils


@struct.dataclass
class PushState:
    """State for the push system, stored in the environment state.info."""
    # Time tracking
    last_push_time: jax.Array  # Last time a push was applied
    next_push_time: jax.Array  # Next scheduled push time
    
    # Current push tracking
    is_pushing: jax.Array  # Boolean: whether currently applying a push
    push_end_time: jax.Array  # When the current push should end
    current_force: jax.Array  # Current 6D wrench being applied (shape: (6,))
    
    # Push generation state
    rng: jax.Array  # RNG state for push generation
    
    @classmethod
    def init(cls, rng: jax.Array, interval_range: jax.Array) -> "PushState":
        """Initialize push state."""
        rng, key = jax.random.split(rng)
        first_push_time = jax.random.uniform(
            key,
            (),
            minval=interval_range[0],
            maxval=interval_range[1]
        )
        return cls(
            last_push_time=jnp.array(-jnp.inf),
            next_push_time=first_push_time,
            is_pushing=jnp.array(False),
            push_end_time=jnp.array(-jnp.inf),
            current_force=jnp.zeros(6),
            rng=rng,
        )


class PushSystem:
    """Handles random external force/torque application for robustness training."""
    
    def __init__(self, push_config: config_dict.ConfigDict, target_body_id: int):
        """
        Initialize the push system.
        
        Args:
            push_config: Configuration dict with push parameters
            target_body_id: MuJoCo body ID to apply pushes to
        """
        self.config = push_config
        self.target_body_id = target_body_id
        
        # Pre-compute some values for efficiency
        self._force_ranges = jnp.array([
            push_config.force_ranges.x,
            push_config.force_ranges.y,
            push_config.force_ranges.z
        ])  # Shape: (3, 2)
        
        self._torque_range = jnp.array(push_config.torque_range)  # Shape: (2,)
        self._interval_range = jnp.array(push_config.interval_range)  # Shape: (2,)
        self._duration_range = jnp.array(push_config.duration_range)  # Shape: (2,)
    
    def update(
        self,
        push_state: PushState,
        current_time: float,
        base_quat: jax.Array,
    ) -> Tuple[PushState, jax.Array]:
        """
        Update push state and return current wrench to apply.
        
        Args:
            push_state: Current push state
            current_time: Current simulation time
            base_quat: Base orientation quaternion for coordinate transformation
            
        Returns:
            Tuple of (updated_push_state, wrench_to_apply)
            wrench_to_apply: 6D array [fx, fy, fz, tx, ty, tz] in world frame
        """
        if not self.config.enabled:
            return push_state, jnp.zeros(6)
        
        # Check if we should start a new push
        should_start_new_push = jnp.logical_and(
            jnp.logical_not(push_state.is_pushing),
            current_time >= push_state.next_push_time
        )
        
        # Check if current push should end
        should_end_current_push = jnp.logical_and(
            push_state.is_pushing,
            current_time >= push_state.push_end_time
        )
        
        # Generate new push if needed
        new_push_state, new_force = lax.cond(
            should_start_new_push,
            self._start_new_push,
            lambda ps, ct, bq: (ps, ps.current_force),
            push_state,
            current_time,
            base_quat,
        )
        
        # End current push if needed
        final_push_state = lax.cond(
            should_end_current_push,
            self._end_current_push,
            lambda ps, ct: ps,
            new_push_state,
            current_time,
        )
        
        # Return appropriate force
        wrench = lax.cond(
            final_push_state.is_pushing,
            lambda: final_push_state.current_force,
            lambda: jnp.zeros(6),
        )
        
        return final_push_state, wrench
    
    def _start_new_push(
        self,
        push_state: PushState,
        current_time: float,
        base_quat: jax.Array,
    ) -> Tuple[PushState, jax.Array]:
        """Start a new push and return updated state and force."""
        rng = push_state.rng
        
        # Sample push parameters
        rng, key = jax.random.split(rng)
        # Sample force magnitudes for each axis
        force_mags = jax.random.uniform(
            key,
            (3,),
            minval=self._force_ranges[:, 0],
            maxval=self._force_ranges[:, 1]
        )
        
        # Sample force directions (random signs)
        rng, key = jax.random.split(rng)
        force_signs = jax.random.choice(key, jnp.array([-1.0, 1.0]), (3,))
        force_body = force_mags * force_signs
        
        # Sample torques
        rng, key = jax.random.split(rng)
        torque_mags = jax.random.uniform(
            key,
            (3,),
            minval=self._torque_range[0],
            maxval=self._torque_range[1]
        )
        rng, key = jax.random.split(rng)
        torque_signs = jax.random.choice(key, jnp.array([-1.0, 1.0]), (3,))
        torque_body = torque_mags * torque_signs
        
        # Transform forces and torques from body frame to world frame
        # (since our push config is specified in body frame but xfrc_applied expects world frame)
        force_world = math_utils.vec_body_to_world(base_quat, force_body)
        torque_world = math_utils.vec_body_to_world(base_quat, torque_body)
        
        # Combine into 6D wrench (force + torque) in world frame
        current_force = jnp.concatenate([force_world, torque_world])
        
        # Sample push duration
        rng, key = jax.random.split(rng)
        push_duration = jax.random.uniform(
            key,
            (),
            minval=self._duration_range[0],
            maxval=self._duration_range[1]
        )
        
        # Schedule next push
        rng, key = jax.random.split(rng)
        next_interval = jax.random.uniform(
            key,
            (),
            minval=self._interval_range[0], 
            maxval=self._interval_range[1]
        )
        
        next_push_time = current_time + push_duration + next_interval
        
        new_state = push_state.replace(
            last_push_time=jnp.array(current_time),
            next_push_time=jnp.array(next_push_time),
            is_pushing=jnp.array(True),
            push_end_time=jnp.array(current_time + push_duration),
            current_force=current_force,
            rng=rng,
        )
        
        return new_state, current_force
    
    def _end_current_push(
        self,
        push_state: PushState,
        current_time: float,
    ) -> PushState:
        """End the current push."""
        return push_state.replace(
            is_pushing=jnp.array(False),
            current_force=jnp.zeros(6),
        )


def apply_wrench_to_body(
    model: mjx.Model,
    data: mjx.Data,
    body_id: int,
    wrench: jax.Array,
) -> mjx.Data:
    """
    Apply a 6D wrench (force + torque) to a specific body.
    
    Args:
        data: MJX data
        body_id: Body ID to apply wrench to
        wrench: 6D array [fx, fy, fz, tx, ty, tz] in world frame
        model: MJX model
        
    Returns:
        Updated MJX data with wrench applied
        
    Note:
        The wrench must be specified in world frame coordinates as expected by xfrc_applied.
    """
    # Build xfrc_applied array - this is in world/global frame
    xfrc = jnp.zeros((model.nbody, 6), dtype=data.xfrc_applied.dtype)
    xfrc = xfrc.at[body_id].set(wrench)
    
    return data.replace(xfrc_applied=xfrc)
