"""Adversary environment for adversarial reinforcement learning with Cassie.

The adversary learns to apply forces to Cassie to make it fall, while Cassie's
policy remains frozen.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import mujoco as mj
import mujoco.mjx as mjx
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

import cassie_arl.cassie_env.math_utils as math_utils
from cassie_arl.cassie_env.push_system import apply_wrench_to_body
from cassie_arl.cassie_env.cassie_env import CassieEnv
from cassie_arl.cassie_env.cassie_consts import QPosIdx, QVelIdx, FOOT_CONTACT_THRESHOLD


def default_adversary_config() -> config_dict.ConfigDict:
    """Default configuration for adversary environment."""
    return config_dict.create(
        # --------------------------------
        # Required simulation parameters
        # --------------------------------
        ctrl_dt=0.01,  # 100 Hz - adversary chooses new forces at this rate
        sim_dt=0.002,  # Match "timestep" in MJCF
        episode_length=800,  # 8 seconds at ctrl_dt=0.01

        # -------------------
        # Adversary parameters
        # -------------------
        # Maximum wrench values [Fx, Fy, Fz, Tx, Ty, Tz]
        # Actions are scaled from [-1, 1] to [-max, +max]
        wrench_max=jnp.array([60.0, 50.0, 60.0, 0.0, 0.0, 0.0]),

        # Target body to apply forces to
        target_body="cassie-pelvis",

        # Reward weights for adversary
        # Adversary gets rewarded for making Cassie fall or move
        adversary_reward_config=config_dict.create(
            weights=config_dict.create(
                cassie_alive=-1.0,  # Negative reward for every step Cassie stays alive (adversary wants Cassie to fall)
                pelvis_tilt=0.4,  # Reward for making pelvis tilt
                pelvis_velocity=0.3,  # Reward for making pelvis move
                energy_penalty=-0.05,  # Small penalty for using force
                force_change_penalty=-0.4,  # Penalty for changing forces/torques too rapidly
            ),
        ),
    )


class AdversaryEnv(mjx_env.MjxEnv):
    """Adversarial environment where the adversary learns to apply forces to Cassie.
    
    The adversary observes Cassie's state and chooses force/torque parameters.
    Cassie's policy is held constant (frozen) during adversary training.
    """
    def __init__(
            self,
            cassie_policy_fn: Callable[[jax.Array, jax.Array], Tuple[jax.Array, Any]],
            xml_path: Optional[str] = None,
            config: config_dict.ConfigDict = default_adversary_config(),
            config_overrides: Optional[Dict[str, Any]] = None,
            cassie_config: Optional[config_dict.ConfigDict] = None,
    ):
        """
        Args:
            cassie_policy_fn: Frozen Cassie policy function (obs, rng) -> (action, extra)
            xml_path: Path to Cassie MJCF file (uses CassieEnv default if None)
            config: Adversary environment configuration
            config_overrides: Optional config overrides
            cassie_config: Optional Cassie environment configuration
        """
        super().__init__(config, config_overrides)
        
        self._cassie_policy_fn = cassie_policy_fn
        
        # Create the Cassie environment that the adversary will interact with

        if cassie_config is None:
            from cassie_arl.cassie_env.cassie_env import default_config
            cassie_config = default_config()
        # Disable push system in Cassie env since adversary controls forces
        cassie_config.push_config.enabled = False
        
        # Match simulation parameters
        cassie_config.ctrl_dt = config.ctrl_dt
        cassie_config.sim_dt = config.sim_dt
        cassie_config.episode_length = config.episode_length
        
        if xml_path is not None:
            self._cassie_env = CassieEnv(xml_path=xml_path, config=cassie_config)
        else:
            self._cassie_env = CassieEnv(config=cassie_config)
        
        self._target_body_id = self._cassie_env._mj_model.body(config.target_body).id
        self._wrench_max = jnp.array(config.wrench_max)  # Shape: (6,)

    # ----------------------------------------------------------------------
    # Required abstract methods/properties
    # ----------------------------------------------------------------------

    @property
    def xml_path(self) -> str:
        return self._cassie_env.xml_path

    @property
    def action_size(self) -> int:
        """Adversary action space: 6 dimensions
        Direct 6D wrench [Fx, Fy, Fz, Tx, Ty, Tz] with values in [-1, 1]
        Each component is scaled to [-max, +max] based on wrench_max config.
        """
        return 6

    @property
    def mj_model(self) -> mj.MjModel:
        return self._cassie_env.mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._cassie_env.mjx_model

    # ----------------------------------------------------------------------
    # Core env logic
    # ----------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment."""
        # Reset Cassie environment
        cassie_state = self._cassie_env.reset(rng)
        
        # Build adversary observation
        obs = self._get_adversary_obs(cassie_state)
        
        info = {
            "rng": rng,
            "step": 0,
            "cassie_state": cassie_state,
            "prev_wrench": jnp.zeros(6),
        }
        
        metrics = {
            "reward": jnp.zeros(()),
            **{
                f"adversary_reward_component/{k}": jnp.zeros(())
                for k in [
                    "cassie_alive",
                    "pelvis_tilt",
                    "pelvis_velocity",
                    "energy_penalty",
                    "force_change_penalty",
                ]
            },
        }
        
        return mjx_env.State(
            data=cassie_state.data,
            obs=obs,
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            metrics=metrics,
            info=info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take one adversary step.
        
        The adversary chooses force parameters, which are applied to Cassie
        for the duration of one control step. Cassie's policy runs as normal.
        
        Args:
            state: Current environment state
            action: Adversary action (6D: force + torque in each axis)
        """
        rng = state.info["rng"]
        cassie_state = state.info["cassie_state"]
        
        wrench = self._action_to_wrench(action, cassie_state.data)
        
        # Get Cassie action from frozen policy
        rng, cassie_rng = jax.random.split(rng)
        cassie_action, _ = self._cassie_policy_fn(cassie_state.obs, cassie_rng)
        
        cassie_state = self._step_cassie_with_force(
            cassie_state, cassie_action, wrench
        )
        
        reward_components = self._get_adversary_reward(
            cassie_state.data,
            wrench,
            state.info.get("prev_wrench", jnp.zeros(6)),
        )
        
        reward_weighted = {
            k: reward_components[k] * self._config.adversary_reward_config.weights[k]
            for k in reward_components
        }
        reward = jnp.sum(jnp.array(list(reward_weighted.values()))) * self.dt
        
        obs = self._get_adversary_obs(cassie_state)
        done = cassie_state.done
        
        new_step = state.info.get("step", 0) + 1
        
        new_info = {
            **state.info,  # Preserve wrapper keys like 'episode_done', 'first_obs', etc.
            "rng": rng,
            "step": new_step,
            "cassie_state": cassie_state,
            "prev_wrench": wrench,
        }
        
        metrics = {
            "reward": reward,
            **{
                f"adversary_reward_component/{k}": v
                for k, v in reward_weighted.items()
            },
        }
        
        return state.replace(
            data=cassie_state.data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=new_info,
        )

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------

    def _get_adversary_obs(self, cassie_state: mjx_env.State) -> jax.Array:
        """Build adversary observation from Cassie's state.
        
        The adversary observes a noiseless subset of Cassie's state:
        - Pelvis position and orientation
        - Pelvis linear and angular velocity
        - Joint positions and velocities
        - Foot contacts
        
        This is similar to Cassie's observation but doesn't include action history.
        """
        data = cassie_state.data
        
        # Import after env is initialized to avoid circular imports
        from cassie_arl.cassie_env.cassie_consts import QPosIdx, QVelIdx
        
        # Base orientation
        base_quat = data.qpos[QPosIdx.BASE_QUAT]
        
        # Yaw-invariant (tilt-only) quaternion
        rpy = math_utils.quat2euler(base_quat)
        tilt_quat = math_utils.euler2quat(jnp.array([rpy[0], rpy[1], 0.0]))
        
        # Pelvis height
        pelvis_height = data.qpos[QPosIdx.BASE_HEIGHT]
        
        # Joint positions and velocities (relative to nominal standing pose)
        motor_qpos = data.qpos[QPosIdx.MOTORS]
        motor_qvel = data.qvel[QVelIdx.MOTORS]
        motor_qpos_delta = motor_qpos - self._cassie_env._standing_jnt_angles
        
        lin_vel_world = data.qvel[QVelIdx.BASE_LIN_VEL]
        ang_vel_body = data.qvel[QVelIdx.BASE_ANG_VEL]
        
        lin_vel_body = math_utils.vec_world_to_body(base_quat, lin_vel_world)
        
        left_foot_contact = self._cassie_env._left_foot_height(data) < FOOT_CONTACT_THRESHOLD
        right_foot_contact = self._cassie_env._right_foot_height(data) < FOOT_CONTACT_THRESHOLD
        feet_contact = jnp.array([left_foot_contact, right_foot_contact], dtype=jnp.float32)
        
        prev_wrench = cassie_state.info.get("prev_wrench", jnp.zeros(6))
        
        obs = jnp.concatenate([
            pelvis_height,
            tilt_quat,
            motor_qpos_delta,
            lin_vel_body,
            ang_vel_body,
            motor_qvel,
            feet_contact,
            prev_wrench,
        ])
        
        return obs

    def _action_to_wrench(self, action: jax.Array, data: mjx.Data) -> jax.Array:
        """Convert adversary action to a 6D wrench in world frame.
        
        Action format (6D):
        - [0:3]: Force components [Fx, Fy, Fz] in body frame, normalized [-1, 1]
        - [3:6]: Torque components [Tx, Ty, Tz] in body frame, normalized [-1, 1]
        
        Each component is scaled by wrench_max and transformed to world frame.
        
        Args:
            action: Adversary action (6D), values in [-1, 1]
            data: MJX data for coordinate transformations
            
        Returns:
            6D wrench [fx, fy, fz, tx, ty, tz] in world frame
        """
        # Import after env is initialized to avoid circular imports
        from cassie_arl.cassie_env.cassie_consts import QPosIdx
        
        action = jnp.clip(action, -1.0, 1.0)
        wrench_body = action * self._wrench_max
        
        force_body = wrench_body[:3]
        torque_body = wrench_body[3:]
        
        # Transform from body frame to world frame
        base_quat = data.qpos[QPosIdx.BASE_QUAT]
        force_world = math_utils.vec_body_to_world(base_quat, force_body)
        torque_world = math_utils.vec_body_to_world(base_quat, torque_body)
        
        wrench = jnp.concatenate([force_world, torque_world])
        return wrench

    def _step_cassie_with_force(
            self,
            cassie_state: mjx_env.State,
            cassie_action: jax.Array,
            wrench: jax.Array,
    ) -> mjx_env.State:
        """Step Cassie environment with adversary force applied.

        Args:
            cassie_state: Current Cassie state
            cassie_action: Action from Cassie's frozen policy
            wrench: 6D wrench to apply in world frame
            
        Returns:
            Updated Cassie state
        """
        pos_targets, p_gains, d_gains = self._cassie_env._parse_action(cassie_action)
        
        # Run PD control substeps with adversary force applied
        def _pd_substep(_: int, carry):
            data_carry, _last_tau = carry
            
            tau = self._cassie_env._pd_control(data_carry, pos_targets, p_gains, d_gains)
            tau = jnp.clip(
                tau,
                self._cassie_env._torque_lowers,
                self._cassie_env._torque_uppers
            )
            
            # Apply adversary force
            data_next = apply_wrench_to_body(
                self.mjx_model,
                data_carry,
                self._target_body_id,
                wrench,
            )
            
            data_next = mjx_env.step(self.mjx_model, data_next, tau, 1)
            return (data_next, tau)
        
        data, torques = lax.fori_loop(
            0,
            self._cassie_env.n_substeps,
            _pd_substep,
            (cassie_state.data, jnp.zeros((self.mjx_model.nu,), dtype=jnp.float32)),
        )
        
        # Build updated adversary observation (privileged)
        # Function expects an RNG but we don't use the noisy observation
        dummy_rng = jax.random.PRNGKey(0) 
        _, privileged_obs_single = self._cassie_env._get_obs(data, torques, dummy_rng)
        obs_single = privileged_obs_single
        
        hist = cassie_state.info["obs_history"]
        privileged_hist = cassie_state.info["privileged_obs_history"]
        
        new_hist = hist.at[1:].set(hist[:-1])
        new_hist = new_hist.at[0].set(obs_single)
        obs = new_hist.reshape(-1)
        
        new_privileged_hist = privileged_hist.at[1:].set(privileged_hist[:-1])
        new_privileged_hist = new_privileged_hist.at[0].set(privileged_obs_single)
        privileged_obs = new_privileged_hist.reshape(-1)
        
        # Compute Cassie's reward (for logging/debugging, not used for adversary)
        per_step_raw = self._cassie_env._get_reward(
            data,
            cassie_state.info,
            pos_targets,
            torques,
            p_gains,
            d_gains,
        )
        per_step_weighted = {
            k: per_step_raw[k] * self._cassie_env._config.reward_config.weights[k]
            for k in per_step_raw
        }
        cassie_reward = jnp.sum(jnp.array(list(per_step_weighted.values()))) * self.dt
        
        # Check termination
        new_step = cassie_state.info.get("step", 0) + 1
        done = self._cassie_env._get_termination(data, jnp.array(new_step))
        done = jnp.array(done, dtype=cassie_reward.dtype)
        
        # Update Cassie state
        new_info = {
            **cassie_state.info,
            "step": new_step,
            "pos_targets": pos_targets,
            "last_p_gains": p_gains,
            "last_d_gains": d_gains,
            "obs_history": new_hist,
            "privileged_obs_history": new_privileged_hist,
        }
        
        metrics = {
            **{f"reward_component/{k}": v for k, v in per_step_weighted.items()},
            "reward": cassie_reward,
        }
        
        return cassie_state.replace(
            data=data,
            obs={"state": obs, "privileged_state": privileged_obs},
            reward=cassie_reward,
            done=done,
            metrics=metrics,
            info=new_info,
        )

    def _get_adversary_reward(
            self,
            data: mjx.Data,
            wrench: jax.Array,
            prev_wrench: jax.Array,
    ) -> Dict[str, jax.Array]:
        """Compute adversary reward components.
        
        The adversary is rewarded for:
        - Making Cassie fall (large negative reward = positive for adversary)
        - Making Cassie deviate from nominal state
        - Penalized slightly for energy usage
        - Penalized for rapid changes in forces/torques
        
        Args:
            data: MJX data
            wrench: Applied wrench (for energy penalty)
            prev_wrench: Previously applied wrench (for change penalty)
            
        Returns:
            Dictionary of reward components (before weighting)
        """
        return {
            "cassie_alive": self._cost_alive(),
            "pelvis_tilt": self._reward_pelvis_tilt(data),
            "pelvis_velocity": self._reward_pelvis_velocity(data),
            "energy_penalty": self._cost_energy(wrench),
            "force_change_penalty": self._cost_force_change(wrench, prev_wrench),
        }

    def _cost_alive(self) -> jax.Array:
        """Cost function for Cassie being alive."""
        return jnp.array(1.0)
    
    def _reward_pelvis_tilt(self, data: mjx.Data) -> jax.Array:
        """Reward function for pelvis tilt.
        
        Encourages the adversary to make Cassie's pelvis tilt away from upright.
        """
        base_quat = data.qpos[QPosIdx.BASE_QUAT]
        rpy = math_utils.quat2euler(base_quat)
        tilt_magnitude = jnp.sqrt(rpy[0]**2 + rpy[1]**2)  # Roll and pitch
        tilt_normalized = jnp.clip(tilt_magnitude / 0.5, 0.0, 1.0)
        return tilt_normalized
    
    def _reward_pelvis_velocity(self, data: mjx.Data) -> jax.Array:
        # Pelvis velocity (encourage movement)
        lin_vel = data.qvel[QVelIdx.BASE_LIN_VEL]
        ang_vel = data.qvel[QVelIdx.BASE_ANG_VEL]
        velocity_magnitude = jnp.sqrt(jnp.sum(lin_vel**2) + jnp.sum(ang_vel**2))
        velocity_normalized = jnp.clip(velocity_magnitude / 2.0, 0.0, 1.0)
        return velocity_normalized

    def _cost_energy(self, wrench: jax.Array) -> jax.Array:
        """Cost function for energy usage, proportional to applied forces/torques.
        
        Penalizes the adversary for applying large forces/torques.
        Encourages adversary to find the minimal force needed to destabilize Cassie.
        """
        epsilon = 1e-8
        force_magnitude = jnp.linalg.norm(wrench[:3])
        torque_magnitude = jnp.linalg.norm(wrench[3:])
        # Normalize by max possible force/torque
        max_force = jnp.linalg.norm(self._wrench_max[:3])
        max_torque = jnp.linalg.norm(self._wrench_max[3:])
        energy = (force_magnitude / (max_force + epsilon) + 
                  torque_magnitude / (max_torque + epsilon)) / 2.0
        energy_normalized = jnp.clip(energy, 0.0, 1.0)

        return energy_normalized

    def _cost_force_change(self, wrench: jax.Array, prev_wrench: jax.Array) -> jax.Array:
        """Cost function for changes in applied forces/torques.
        
        Penalizes rapid changes in force and torque to encourage smoother,
        more realistic forces. The cost is timestep-invariant and
        scaled such that traversing the full force/torque range in 1 second
        incurs a cost of 1.0.
        
        Args:
            wrench: Current wrench (6D: [fx, fy, fz, tx, ty, tz])
            prev_wrench: Previous wrench from last step
            
        Returns:
            Normalized cost in [0, 1] based on rate of change
        """
        epsilon = 1e-8
        
        force_change = jnp.linalg.norm(wrench[:3] - prev_wrench[:3])
        torque_change = jnp.linalg.norm(wrench[3:] - prev_wrench[3:])
        
        # Normalize by max possible change (from -max to +max)
        max_force = jnp.linalg.norm(self._wrench_max[:3])
        max_torque = jnp.linalg.norm(self._wrench_max[3:])
        
        # Max change is 2 * max (from -max to +max)
        force_change_normalized = force_change / (2.0 * max_force + epsilon)
        torque_change_normalized = torque_change / (2.0 * max_torque + epsilon)
        
        # Average the two components
        change = (force_change_normalized + torque_change_normalized) / 2.0
        
        # Make timestep-invariant by converting to rate of change per second
        change_per_second = change / self.dt
        
        return jnp.clip(change_per_second, 0.0, 1.0)
    