from typing import Any, Dict, Optional, Union
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
from flax import struct
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

from cassie_arl.config.cassie_consts import *
import cassie_arl.rl_env.math_utils as math_utils
from cassie_arl.rl_env.butterworth import BiquadFilter, design_butterworth_biquad


script_dir = Path(__file__).parent
CASSIE_SCENE_XML = script_dir.parent / "models" / "scene.xml"


@struct.dataclass
class RewardComponents:
    """PyTree for per-step reward components kept in state.info.

    Storing rewards in a dataclass makes the structure static and JIT-friendly
    (vs an arbitrary dict). All leaves are jax.Arrays with scalar shape ().
    """
    alive: jax.Array
    pelvis_lin_vel: jax.Array
    pelvis_tilt: jax.Array
    motor_ref_error: jax.Array
    action_rate: jax.Array
    torques: jax.Array

    @classmethod
    def zeros(cls) -> "RewardComponents":
        z = jnp.zeros(())
        return cls(
            alive=z,
            pelvis_lin_vel=z,
            pelvis_tilt=z,
            motor_ref_error=z,
            action_rate=z,
            torques=z,
        )



def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        # --------------------------------
        # Required simulation parameters
        # --------------------------------
        ctrl_dt=0.02,
        sim_dt=0.002,  # Match "timestep" in MJCF
        episode_length=500,  # 10 seconds at ctrl_dt=0.02
        history_len=1,

        # -------------------
        # Custom parameters
        # -------------------

        soft_joint_pos_limit_factors=0.95,  # Fraction of full range to use as soft limits on actuated joints

        # PD Gains
        # p_gain = jnp.array([
        #     8, 4, 4, 10, 0.4,
        #     8, 4, 4, 10, 0.4
        # ]),
        # d_gain = jnp.array([
        #     0.08, 0.08, 0.2, 0.4, 0.08,
        #     0.08, 0.08, 0.2, 0.4, 0.08
        # ]),
        p_gain = jnp.array([
            8, 4, 4, 10, 0.4,
            8, 4, 4, 10, 0.4
        ]) / 5,
        d_gain = jnp.array([
            0.08, 0.08, 0.2, 0.4, 0.08,
            0.08, 0.08, 0.2, 0.4, 0.08
        ]) / 5,


        # Reward function configuration
        # Except for the "fall" cost, which is a one-time cost, all reward weights
        # are in [0, 1] and all cost weights are in [-1, 0].
        reward_config=config_dict.create(
            scales=config_dict.create(
                alive=2.0,
                pelvis_lin_vel=-0.7,
                pelvis_tilt=-0.6,
                motor_ref_error=-0.7,
                action_rate=-0.6,
                torques=-0.3,
            ),
        ),

        filters=config_dict.create(
            vel_cutoff_hz=25.0,    # Hz for measured joint velocity filtering
            target_cutoff_hz=10.0, # Hz for target joint angle smoothing
        )
    )


class CassieEnv(mjx_env.MjxEnv):
    """Cassie environment built on MJX, compatible with Brax PPO."""
    # TODO: add random pushes to improve robustness
    # TODO: add metrics

    def __init__(
            self,
            xml_path: str = CASSIE_SCENE_XML.as_posix(),
            config: config_dict.ConfigDict = default_config(),
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None
    ):
        super().__init__(config, config_overrides)
        # Load MuJoCo model and MJX version
        self._xml_path = xml_path
        self._mj_model = mj.MjModel.from_xml_path(self._xml_path)
        self._mjx_model = mjx.put_model(self._mj_model)

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._post_init()

    def _post_init(self):
        self._init_qpos = jnp.array(self._mj_model.keyframe("home").qpos)
        self._standing_jnt_angles = self._init_qpos[QPosIdx.MOTORS]
        # self._standing_torques = jnp.array(self._mj_model.keyframe("home").ctrl)
        self._standing_torques = jnp.array([0, 0, -0.358, 2.4205, 0, 0, 0, -0.358, 2.4205, 0])  # Knee value too low => robot goes up; hip pitch too negative => pelvis tilts backwards

        standing_quat = self._init_qpos[QPosIdx.BASE_QUAT]
        self._standing_base_rpy = math_utils.quat2euler(standing_quat)

        # Apply soft limits on actuated joints for safe hardware deployment
        self._jnt_lowers = jnp.array(self._mj_model.jnt_range[JntRangeIdx.MOTORS, 0])
        self._jnt_uppers = jnp.array(self._mj_model.jnt_range[JntRangeIdx.MOTORS, 1])
        jnt_c = (self._jnt_lowers + self._jnt_uppers) / 2
        jnt_r = self._jnt_uppers - self._jnt_lowers
        self._jnt_soft_lowers = jnt_c - 0.5 * jnt_r * self._config.soft_joint_pos_limit_factors
        self._jnt_soft_uppers = jnt_c + 0.5 * jnt_r * self._config.soft_joint_pos_limit_factors

        self._torque_lowers = jnp.array(self._mj_model.actuator_ctrlrange[:, 0])
        self._torque_uppers = jnp.array(self._mj_model.actuator_ctrlrange[:, 1])

        def geoms_of_body(model, body_id):
            start = model.body_geomadr[body_id]
            count = model.body_geomnum[body_id]
            geom_ids = jnp.arange(start, start + count)
            return geom_ids
        
        # MJCF geom and body IDs
        self._floor_gid = self._mj_model.geom("floor").id
        self._pelvis_id = self._mj_model.body("cassie-pelvis").id
        self._left_foot_id = self._mj_model.body("left-foot").id
        self._right_foot_id = self._mj_model.body("right-foot").id
        self._left_tarsus_id = self._mj_model.body("left-tarsus").id
        self._right_tarsus_id = self._mj_model.body("right-tarsus").id

        self._left_foot_gid = geoms_of_body(self._mj_model, self._left_foot_id)
        self._right_foot_gid = geoms_of_body(self._mj_model, self._right_foot_id)

        # PD gains
        self._p_gain = self._config.p_gain
        self._d_gain = self._config.d_gain

        # -------------------
        # Butterworth filter setup
        # -------------------
        self._ctrl_rate_hz = 1.0 / self.dt

        # Butterworth filter for qvel
        self._vel_filter = BiquadFilter.create(
            design_butterworth_biquad(
                self._config.filters.vel_cutoff_hz,
                self._ctrl_rate_hz
            ),
            self.action_size
        )

        # Butterworth filter for target joint angles
        self._target_filter = BiquadFilter.create(
            design_butterworth_biquad(
                self._config.filters.target_cutoff_hz,
                self._ctrl_rate_hz
            ),
            self.action_size
        )

    # ----------------------------------------------------------------------
    # Required abstract methods/properties
    # ----------------------------------------------------------------------

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        # Cassie has 10 actuators (5 per leg)
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mj.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # ----------------------------------------------------------------------
    # Core env logic
    # ----------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resets Cassie to default pose + small random perturbations."""
        qpos = self._init_qpos
        qvel = jnp.zeros(self._mjx_model.nv)

        data = mjx_env.init(self._mjx_model, qpos=qpos, qvel=qvel)
        obs = self._get_obs(data)

        info = {
            "rng": rng,
            "step": 0,
            # Keep reward components as a PyTree for JIT friendliness.
            "reward_components": RewardComponents.zeros(),
            "last_action": jnp.zeros((self.action_size,)),
            "last_torques": jnp.zeros((self.action_size,)),
            # Filter states
            "vel_filter": self._vel_filter,
            "target_filter": self._target_filter,
        }

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            metrics={},
            info=info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """
        Takes one control step in Cassie.

        Args:
            state: Current environment state.
            action: Action to take. Shape: [action_size], values in [-1, 1].
                    Should be interpreted as normalized deltas for
                    the 10 actuated joints.
        """
        rng = state.info["rng"]

        action = jnp.clip(action, -1.0, 1.0)

        # 1. Convert action to raw joint position targets (vector length 10)
        raw_pos_targets = self._action_to_jnt_targets(action)

        # 2. Filter targets (for smoother commanded motion)
        target_filter = state.info["target_filter"]
        filtered_targets, new_target_filter = target_filter.apply(raw_pos_targets)

        # 3. PD control with filtered velocities
        # We'll pass filter state so _pd_control can update it.
        torques, new_vel_filter = self._pd_control(
            state.data,
            filtered_targets,
            self._p_gain,
            self._d_gain,
            state.info["vel_filter"],
        )
        torques = self._limit_torque_rate(torques, state.info["last_torques"])
        torques = jnp.clip(torques, self._torque_lowers, self._torque_uppers)

        data = mjx_env.step(
            self._mjx_model, state.data, torques, self.n_substeps
        )

        obs = self._get_obs(data)

        # Get reward components. `_get_reward` returns ((per_step_raw, event_raw), (lift_given_new, current_both_feet, current_com_error))
        # (per_step_raw, event_raw), (lift_given_new, current_both_feet, current_com_error) = self._get_reward(data, action, state.info)
        per_step_raw = self._get_reward(
                            data,
                            action,
                            state.info,
                            torques
                        )

        # Scale each component by its configured weight
        per_step_scaled = {k: per_step_raw[k] * self._config.reward_config.scales[k] for k in per_step_raw}

        # Per-step components are integrated over dt; event components are added directly
        reward = sum(per_step_scaled.values()) * self.dt
        reward = jnp.clip(reward, -1000, 1000)

        new_step = state.info.get("step", 0) + 1
        
        done = self._get_termination(data, jnp.array(new_step))
        done = jnp.array(done, dtype=reward.dtype)

        new_info = {
            **state.info,
            "step": new_step,
            "rng": rng,
            # Store reward components as a PyTree for stability under JIT.
            "reward_components": RewardComponents(**per_step_scaled),
            "last_action": action,  # For action rate cost
            "last_torques": torques,  # For torque rate limiting
            "vel_filter": new_vel_filter,
            "target_filter": new_target_filter,
        }
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            info=new_info
        )
    
    def _get_obs(self, data) -> jax.Array:
        """Constructs observation from mjx.Data (Cassie)."""
        # TODO: separate "state" and "privileged state"
        # TODO: We are now using joint angle deltas. Consider changing the action
        #       to represent deltas relative to standing pose as well.
        # TODO: consider adding foot forces
        # TODO: consider adding dual history architecture (Z Li). Probably not all
        #       that useful right now because state is relatively Markovian. But may
        #       be useful later when noise and external pushes are added.

        motor_qpos = data.qpos[QPosIdx.MOTORS]
        motor_qvel = data.qvel[QVelIdx.MOTORS]
        pelvis_qvel = data.qvel[QVelIdx.BASE]

        # Add pelvis orientation as a flat quaternion
        pelvis_quat = data.qpos[QPosIdx.BASE_QUAT]

        # Pelvis height (z coord of root)
        pelvis_height = data.qpos[QPosIdx.BASE_HEIGHT]  # shape (1,)

        # Use difference between current motor angles and the standing pose
        motor_qpos_delta = motor_qpos - self._standing_jnt_angles

        # Concatenate into a single vector. Order chosen to keep base-state first,
        # then motor errors and velocities, then pelvis vel, then foot contact and com->support.
        obs = jnp.concatenate([
            pelvis_height,
            pelvis_quat,
            motor_qpos_delta,
            pelvis_qvel,
            motor_qvel
        ])

        return obs

    def _get_reward(
            self,
            data: mjx.Data,
            action: jax.Array,
            info: Dict[str, Any],
            torques: jax.Array
        ) -> Dict[str, jax.Array]:
        """
        Computes reward components.

        All rewards/costs are in [0, 1]. Their weights in self._config.reward_config.scales
        determine their relative importance and sign.
        """
        # TODO: IMPORTANT: need to reward active recovery strategies, not just standing still
        # TODO: look into selective/adaptive rewards e.g. lift costs for movement when perturbing robot

        # Split components into per-step (integrated over dt) and event (one-time) rewards.
        per_step = {
            "alive": self._reward_alive(),
            "pelvis_lin_vel": self._cost_pelvis_lin_vel(data),
            "pelvis_tilt": self._cost_pelvis_tilt(data),
            "motor_ref_error": self._cost_motor_reference_error(data),
            "action_rate": self._cost_action_rate(action, info["last_action"]),
            "torques": self._cost_torques(torques),
        }

        return per_step

    def _reward_alive(self) -> jax.Array:
        """Reward for staying 'alive' (not falling over)."""
        return jnp.array(1.0)
    
    def _cost_pelvis_lin_vel(self, data: mjx.Data) -> jax.Array:
        """Cost for pelvis linear velocity."""
        pelvis_lin_vel = data.qvel[QVelIdx.BASE_LIN_VEL]
        v_sq = jnp.mean(pelvis_lin_vel**2)
        v_scale = 0.8**2  # Normalizing constant (m/s)^2
        cost = v_sq / v_scale
        return jnp.clip(cost, 0.0, 1.0)

    def _cost_pelvis_tilt(self, data: mjx.Data) -> jax.Array:
        """Cost for pelvis orientation (deviation from standing)."""
        # Get base quaternion
        base_quat = data.qpos[QPosIdx.BASE_QUAT]
        rpy = math_utils.quat2euler(base_quat)

        # Only roll and pitch
        orientation_err = math_utils.angle_diff(rpy[:2], self._standing_base_rpy[:2])

        # Mean squared error, normalized
        err_scale = 0.35  # radians (~20 degrees)
        orientation_cost = jnp.mean((orientation_err / err_scale) ** 2)

        # Clip to [0,1]
        return jnp.clip(orientation_cost, 0.0, 1.0)

    def _cost_motor_reference_error(self, data: mjx.Data) -> jax.Array:
        """Cost for deviation of the motor angles from reference standing pose."""
        motor_qpos = data.qpos[QPosIdx.MOTORS]
        err = motor_qpos - self._standing_jnt_angles
        err_scale = 0.35  # Normalizing constant (radians)
        cost = jnp.mean((err / err_scale)**2)
        return jnp.clip(cost, 0.0, 1.0)

    def _cost_action_rate(self, action: jax.Array, last_action: jax.Array) -> jax.Array:
        """Cost for large changes in action between steps."""
        # If action moves through the full range in quarter of a second, incur max cost.
        act_rate = action - last_action
        rate_scale = 8.0 * self.dt  # Normalizing constant
        cost = jnp.mean((act_rate / rate_scale)**2)
        return jnp.clip(cost, 0.0, 1.0)
    
    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        """Cost for large torques."""
        # Incur max cost if using max torque
        torque_scales = self._torque_uppers  # Normalizing constant
        cost = jnp.mean((torques / torque_scales)**2)
        return jnp.clip(cost, 0.0, 1.0)

    def _get_termination(self, data: mjx.Data, step: jax.Array) -> jax.Array:
        """Return True if Cassie has fallen or max timesteps reached."""
        fallen = self._has_fallen(data)

        max_steps = jnp.array(self._config.episode_length, dtype=step.dtype)
        max_steps_reached = step >= max_steps

        done = jnp.logical_or(fallen, max_steps_reached)
        return done

    def _action_norm2torque(
            self,
            action: jax.Array,
            torque_lb: jax.Array,
            torque_ub: jax.Array,
        ) -> jax.Array:
        """
        Converts normalized action in [-1, 1] to torques.
        """
        # Scale action to torque range
        return torque_lb + 0.5 * (torque_ub - torque_lb) * (action + 1.0)
    
    def _has_fallen(self, data: mjx.Data) -> jax.Array:
        """Returns True if Cassie has fallen (pelvis height below threshold or tarsus hit ground)."""
        pelvis_fallen = data.qpos[QPosIdx.BASE_HEIGHT].squeeze() < FALLING_THRESHOLD
        tarsus_hit = self._tarsus_hit_ground(data)
        return jnp.logical_or(pelvis_fallen, tarsus_hit)

    def _tarsus_hit_ground(self, data: mjx.Data) -> jax.Array:
        """Returns True if either tarsus has hit the ground (below threshold)."""
        left_tarsus_z = data.xpos[self._left_tarsus_id, 2]
        right_tarsus_z = data.xpos[self._right_tarsus_id, 2]
        
        left_hit = left_tarsus_z < TARSUS_HIT_GROUND_THRESHOLD
        right_hit = right_tarsus_z < TARSUS_HIT_GROUND_THRESHOLD
        
        return jnp.logical_or(left_hit, right_hit)

    def _action_to_jnt_targets(self, action: jax.Array) -> jax.Array:
        """
        Scales normalized actions [-1, 1] to motor joint angles.

        The action is interpreted as normalized deltas of the 10 actuated
        joints from the standing pose angles. The scaling consists of a
        piecewise linear mapping:
            [-1, 0] -> [soft lower joint limit, standing angle]
            [ 0, 1] -> [standing angle, soft upper joint limit]
        """
        s_pos = self._jnt_soft_uppers - self._standing_jnt_angles   # Positive side span
        s_neg = self._standing_jnt_angles - self._jnt_soft_lowers   # Negative side span
        return self._standing_jnt_angles + 0.5 * (s_pos + s_neg) * action + 0.5 * (s_pos - s_neg) * jnp.abs(action)

    def _pd_control(
            self,
            data: mjx.Data,
            pos_targets: jax.Array,
            p_gain: jax.Array,
            d_gain: jax.Array,
        vel_filter_state: BiquadFilter | None,
    ) -> jax.Array:
        """Computes PD control torques with optional velocity low-pass filtering.

        Returns tuple (torques, new_vel_filter_state) if velocity filter enabled, else (torques, vel_filter_state).
        """
        motor_qpos = data.qpos[QPosIdx.MOTORS]
        raw_motor_qvel = data.qvel[QVelIdx.MOTORS]

        # Filter the noisy velocity
        filt_vel, new_state = vel_filter_state.apply(raw_motor_qvel)

        pos_error = pos_targets - motor_qpos
        vel_error = -filt_vel  # Desire zero vel

        torques = p_gain * pos_error + d_gain * vel_error + self._standing_torques
        return torques, new_state
    
    def _limit_torque_rate(
            self,
            torques: jax.Array,
            last_torques: jax.Array
        ) -> jax.Array:
        """Clips torques if changes by more than the max torque rate."""
        max_torque_rate = (self._torque_uppers - self._torque_lowers) * 2  # Moves through the full range in 0.5 seconds
        max_delta = max_torque_rate * self.dt
        return jnp.clip(torques, last_torques - max_delta, last_torques + max_delta)
    