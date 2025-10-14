"""Push system for applying external disturbances to Cassie during training."""

from typing import Callable, Optional, Tuple, Any
import jax
import jax.numpy as jnp
from jax import lax
import mujoco.mjx as mjx
from flax import struct
from ml_collections import config_dict

import cassie_arl.cassie_env.math_utils as math_utils


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
    
    # Push generation RNG state
    rng: jax.Array
    
    @classmethod
    def init(cls, rng: jax.Array, interval_range: jax.Array) -> "PushState":
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
    """Generates random external force/torque application for testing purposes."""
    
    def __init__(self, push_config: config_dict.ConfigDict, target_body_id: int):
        """
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
        """Update push state and return current wrench to apply.
        
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
        
        should_start_new_push = jnp.logical_and(
            jnp.logical_not(push_state.is_pushing),
            current_time >= push_state.next_push_time
        )
        
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
        
        rng, key = jax.random.split(rng)
        force_mags = jax.random.uniform(
            key,
            (3,),
            minval=self._force_ranges[:, 0],
            maxval=self._force_ranges[:, 1]
        )
        
        rng, key = jax.random.split(rng)
        force_signs = jax.random.choice(key, jnp.array([-1.0, 1.0]), (3,))
        force_body = force_mags * force_signs
        
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


@struct.dataclass
class AdversaryState:
    """State for the adversary system."""
    rng: jax.Array  # RNG state for adversary policy
    
    @classmethod
    def init(cls, rng: jax.Array) -> "AdversaryState":
        """Initialize adversary state."""
        return cls(rng=rng)


class AdversarySystem:
    """Handles adversarial wrench application using a frozen adversary policy."""
    
    def __init__(
        self,
        adversary_policy_fn: Optional[Callable[[jax.Array, jax.Array], Tuple[jax.Array, Any]]],
        target_body_id: int,
        left_foot_id: int,
        right_foot_id: int,
        enabled: bool = True,
    ):
        """
        Args:
            adversary_policy_fn: Frozen adversary policy (obs, rng) -> (action, extra).
                                 If None, system is disabled.
            target_body_id: MuJoCo body ID to apply wrenches to
            left_foot_id: Left foot body ID for contact detection
            right_foot_id: Right foot body ID for contact detection
            enabled: Whether adversary system is enabled
        """
        self.policy_fn = adversary_policy_fn
        self.target_body_id = target_body_id
        self.left_foot_id = left_foot_id
        self.right_foot_id = right_foot_id
        self.enabled = enabled and (adversary_policy_fn is not None)
    
    def get_observation(self, data: mjx.Data, standing_jnt_angles: jax.Array) -> jax.Array:
        """
        Build adversary observation from Cassie's state.
        
        This matches the observation structure in AdversaryEnv._get_adversary_obs.
        
        Args:
            data: MJX data
            standing_jnt_angles: Standing joint angles for computing deltas
            
        Returns:
            Adversary observation array
        """
        from cassie_arl.cassie_env.cassie_consts import (
            QPosIdx, QVelIdx, FOOT_OFFSET, FOOT_CONTACT_THRESHOLD
        )
        
        # Base orientation
        base_quat = data.qpos[QPosIdx.BASE_QUAT]
        
        # Yaw-invariant (tilt-only) quaternion
        rpy = math_utils.quat2euler(base_quat)
        tilt_quat = math_utils.euler2quat(jnp.array([rpy[0], rpy[1], 0.0]))
        
        pelvis_height = data.qpos[QPosIdx.BASE_HEIGHT]
        
        # Joint positions and velocities (relative to standing pose)
        motor_qpos = data.qpos[QPosIdx.MOTORS]
        motor_qvel = data.qvel[QVelIdx.MOTORS]
        motor_qpos_delta = motor_qpos - standing_jnt_angles
        
        lin_vel_world = data.qvel[QVelIdx.BASE_LIN_VEL]
        ang_vel_body = data.qvel[QVelIdx.BASE_ANG_VEL]
        
        lin_vel_body = math_utils.vec_world_to_body(base_quat, lin_vel_world)
        
        left_foot_height = data.xpos[self.left_foot_id, 2] - FOOT_OFFSET
        right_foot_height = data.xpos[self.right_foot_id, 2] - FOOT_OFFSET
        left_foot_contact = left_foot_height < FOOT_CONTACT_THRESHOLD
        right_foot_contact = right_foot_height < FOOT_CONTACT_THRESHOLD
        feet_contact = jnp.array([left_foot_contact, right_foot_contact], dtype=jnp.float32)
        
        obs = jnp.concatenate([
            pelvis_height,
            tilt_quat,
            motor_qpos_delta,
            lin_vel_body,
            ang_vel_body,
            motor_qvel,
            feet_contact,
        ])
        
        return obs
    
    def update(
        self,
        adversary_state: AdversaryState,
        data: mjx.Data,
        standing_jnt_angles: jax.Array,
    ) -> Tuple[AdversaryState, jax.Array]:
        """Update adversary state and return wrench to apply.
        
        Args:
            adversary_state: Current adversary state
            data: MJX data for building observation
            standing_jnt_angles: Standing joint angles
            
        Returns:
            Tuple of (updated_adversary_state, wrench_to_apply)
            wrench_to_apply: 6D array [fx, fy, fz, tx, ty, tz] in world frame
        """
        if not self.enabled:
            return adversary_state, jnp.zeros(6)
        
        obs = self.get_observation(data, standing_jnt_angles)
        
        rng, policy_rng = jax.random.split(adversary_state.rng)
        action, _ = self.policy_fn(obs, policy_rng)
        
        # Action is 6D wrench in normalized space [-1, 1]
        # It needs to be converted to world frame wrench
        from cassie_arl.cassie_env.cassie_consts import QPosIdx
        
        # Default wrench_max values (should match adversary training config)
        wrench_max = jnp.array([50.0, 40.0, 20.0, 10.0, 10.0, 10.0])
        
        action = jnp.clip(action, -1.0, 1.0)
        wrench_body = action * wrench_max
        
        force_body = wrench_body[:3]
        torque_body = wrench_body[3:]
        
        # Transform from body frame to world frame
        base_quat = data.qpos[QPosIdx.BASE_QUAT]
        force_world = math_utils.vec_body_to_world(base_quat, force_body)
        torque_world = math_utils.vec_body_to_world(base_quat, torque_body)
        
        wrench = jnp.concatenate([force_world, torque_world])
        new_state = adversary_state.replace(rng=rng)
        return new_state, wrench
