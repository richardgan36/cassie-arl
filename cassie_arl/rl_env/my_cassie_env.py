from typing import Any, Dict, Optional, Union
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
from absl import logging
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

from cassie_arl.config.cassie_consts import *
import cassie_arl.rl_env.math_utils as math_utils


script_dir = Path(__file__).parent
CASSIE_SCENE_XML = script_dir / ".." / "models" / "scene.xml"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        # --------------------------------
        # Required simulation parameters
        # --------------------------------
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=500,  # 10 seconds at ctrl_dt=0.02
        action_repeat=1,
        history_len=1,

        # -------------------
        # Custom parameters
        # -------------------

        # PD gains for the 10 actuated joints
        # Values are from Z Li et al. 2024 but scaled down to avoid excessive saturation
        p_gain = jnp.array([
            8, 4, 4, 10, 0.4,
            8, 4, 4, 10, 0.4
        ]),
        d_gain = jnp.array([
            0.08, 0.08, 0.2, 0.4, 0.08,
            0.08, 0.08, 0.2, 0.0, 0.08
        ]),
        pd_uncertainty=0.1,  # ±10% uniform randomization of PD gains per episode
        
        # Soft joint limits as a fraction of total joint range
        # (1.0 = full range, 0.0 = no movement allowed)
        soft_joint_pos_limit_factors=jnp.array([
            0.5, 0.8, 0.95, 0.95, 0.95,  # L_HIP_ROLL, L_HIP_YAW, L_HIP_PITCH, L_KNEE, L_FOOT
            0.5, 0.8, 0.95, 0.95, 0.95   # R_HIP_ROLL, R_HIP_YAW, R_HIP_PITCH, R_KNEE, R_FOOT
        ]),
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(  # TODO: define scale of noise for each joint
                gyro=0.2,  # angvel.
            ),
        ),

        # Reward function configuration
        # Except for the "fall" cost, which is a one-time cost, all reward weights
        # are in [0, 1] and all cost weights are in [-1, 0].
        reward_config=config_dict.create(
            scales=config_dict.create(
                alive=1.0,
                fall=-5.0,
                pelvis_lin_vel=-0.2,
                pelvis_tilt=-0.1,
                motor_ref_error=-0.2
            ),
        ),
    )


class CassieEnv(mjx_env.MjxEnv):
    """Cassie environment built on MJX, compatible with Brax PPO."""
    # TODO: replace manual rendering in train_cassie with render method for better debugging
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
        self._default_pose = jnp.array(self._mj_model.keyframe("home").qpos[7:])

        # Apply soft limits on actuated joints for safe hardware deployment
        self._jnt_lowers, self._jnt_uppers = self._mj_model.jnt_range[JntRangeIdx.MOTORS].T
        jnt_c = (self._jnt_lowers + self._jnt_uppers) / 2
        jnt_r = self._jnt_uppers - self._jnt_lowers
        self._jnt_soft_lowers = jnt_c - 0.5 * jnt_r * self._config.soft_joint_pos_limit_factors
        self._jnt_soft_uppers = jnt_c + 0.5 * jnt_r * self._config.soft_joint_pos_limit_factors

        self._torque_lowers, self._torque_uppers = self._mj_model.actuator_ctrlrange.T

        # self._pelvis_id = self._mj_model.body("cassie-pelvis").id

        self._p_gain = self._config.p_gain
        self._d_gain = self._config.d_gain

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

        rng, qpos, qvel = self._add_perturbations(rng, qpos, qvel)
        rng, p_gain, d_gain = self._randomize_pd_gain(rng)

        data = mjx_env.init(self._mjx_model, qpos=qpos, qvel=qvel)
        obs = self._get_obs(data)

        info = {
            "rng": rng,
            "step": 0,
            "p_gain": p_gain,
            "d_gain": d_gain,
            "reward_components": {
                "alive": jnp.zeros(()),
                "fall": jnp.zeros(()),
                "pelvis_lin_vel": jnp.zeros(()),
                "pelvis_tilt": jnp.zeros(()),
                "motor_ref_error": jnp.zeros(()),
            },
            "action": jnp.zeros((self.action_size,)), 
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
                    Should be interpreted as normalized position targets for
                    the 10 actuated joints.
        """
        rng = state.info["rng"]
        rng, key = jax.random.split(rng)

        # jax.debug.print("action: {}", action)

        pos_targets = self._action_norm2actual(action)

        # jax.debug.print("pos_targets: {}", pos_targets)

        p_gain = state.info["p_gain"]
        d_gain = state.info["d_gain"]

        torques = self._pd_control(
            state.data,
            pos_targets,
            p_gain,
            d_gain,
            self._torque_lowers,
            self._torque_uppers
        )
        # jax.debug.print("torques: {}", torques)

        data = state.data
        data = mjx_env.step(
            self._mjx_model, state.data, torques, self.n_substeps
        )

        # jax.debug.print("data.contact: {}", data.contact)

        obs = self._get_obs(data)

        rewards = self._get_reward(data, action)

        # for k, v in rewards.items():
        #     jax.debug.print("{}: {}", k, v)
            
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = sum(rewards.values()) * self.dt
        reward = jnp.clip(reward, -1000, 1000)

        # jax.debug.print("scalar reward: {}", reward)

        new_step = state.info.get("step", 0) + 1
        
        done = self._get_termination(data, jnp.array(new_step))
        done = jnp.array(done, dtype=reward.dtype)

        # new_info = {**state.info, "step": new_step, "rng": rng}
        new_info = {
            **state.info,
            "step": new_step,
            "rng": rng,
            "reward_components": rewards,  # For debugging
            "action": action               # For debugging
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
        # TODO: add foot contact info

        # qpos: joint + base positions
        # qvel: joint + base velocities
        # First 7 entries of qpos are free joint (base pos + quaternion)
        motor_qpos = data.qpos[QPosIdx.MOTORS]
        motor_qvel = data.qvel[QVelIdx.MOTORS]
        pelvis_qvel = data.qvel[QVelIdx.BASE]

        # Add pelvis orientation as a flat quaternion
        pelvis_quat = data.qpos[QPosIdx.BASE_QUAT]
        gravity_in_pelvis_frame = math_utils.gravity_in_base_frame(pelvis_quat)

        # Pelvis height (z coord of root)
        pelvis_height = data.qpos[QPosIdx.BASE_HEIGHT]  # shape (1,)

        # Concatenate into a single vector
        obs = jnp.concatenate([
            pelvis_height,
            pelvis_quat,
            gravity_in_pelvis_frame,
            motor_qpos,
            motor_qvel,
            pelvis_qvel
        ])

        return obs

    def _get_reward(self, data: mjx.Data, action: jax.Array) -> dict[str, jax.Array]:
        """
        Computes reward components.

        All rewards/costs are in [0, 1]. Their weights in self._config.reward_config.scales
        determine their relative importance and sign.
        """
        # TODO: IMPORTANT: need to reward active recovery strategies, not just standing still
        # TODO: look into selective/adaptive rewards e.g. lift costs for movement when perturbing robot
        # TODO: reward for COM above support polygon
        # TODO: cost for large change in acceleration
        # TODO: cost for large torques
        return {
            "alive": self._reward_alive(data),
            "fall": self._cost_fall(data),
            "pelvis_lin_vel": self._cost_pelvis_lin_vel(data),
            "pelvis_tilt": self._cost_pelvis_tilt(data),
            "motor_ref_error": self._cost_motor_reference_error(data),
        }

    def _reward_alive(self, data: mjx.Data) -> jax.Array:
        """Reward for staying 'alive' (not falling over)."""
        return jnp.array(1.0)
    
    def _cost_fall(self, data: mjx.Data) -> jax.Array:
        """One time cost for falling over."""
        fallen = self._has_fallen(data)
        return jnp.where(fallen, 1.0, 0.0)

    def _cost_pelvis_lin_vel(self, data: mjx.Data) -> jax.Array:
        """Cost for pelvis linear velocity."""
        pelvis_lin_vel = data.qvel[QVelIdx.BASE_LIN_VEL]
        v_sq = jnp.mean(pelvis_lin_vel**2)
        v_scale = 0.8**2  # Normalizing constant (m/s)^2
        cost = v_sq / v_scale
        return jnp.clip(cost, 0, 1)

    def _cost_pelvis_tilt(self, data: mjx.Data) -> jax.Array:
        """Cost for pelvis orientation (deviation from standing)."""
        # Get pelvis quaternion
        pelvis_quat = data.qpos[QPosIdx.BASE_QUAT]
        rpy = math_utils.quat2euler(pelvis_quat)

        # Only roll and pitch
        orientation_err = math_utils.angle_diff(rpy[:2], StandingPose.PELVIS_RPY[:2])

        # Mean squared error, normalized
        err_scale = 0.26  # radians (~15 degrees)
        orientation_cost = jnp.mean((orientation_err / err_scale) ** 2)

        # Clip to [0,1]
        return jnp.clip(orientation_cost, 0.0, 1.0)

    def _cost_motor_reference_error(self, data: mjx.Data) -> jax.Array:
        """Cost for deviation of the motor angles from reference standing pose."""
        motor_qpos = data.qpos[QPosIdx.MOTORS]
        err = motor_qpos - StandingPose.MOTOR_ANGLES
        err_scale = 0.35  # Normalizing constant (radians)
        cost = jnp.mean((err / err_scale)**2)
        return jnp.clip(cost, 0, 1)

    def _get_termination(self, data: mjx.Data, step: jax.Array) -> jax.Array:
        """Return True if Cassie has fallen or max timesteps reached."""
        # Pelvis height (z-coordinate)
        fallen = self._has_fallen(data)

        max_steps = jnp.array(self._config.episode_length, dtype=step.dtype)
        max_steps_reached = step >= max_steps

        # Approximate tarsus (toe) positions
        # For simplicity, assume fixed offsets from pelvis for standing
        # Left and right toe z positions
        # left_toe_z = pelvis_z - 1.0  # adjust based on leg length
        # right_toe_z = pelvis_z - 1.0

        # toe_hit_ground = jnp.any(jnp.array([left_toe_z, right_toe_z]) <= TARSUS_HITGROUND_THRESHOLD)

        # return fallen | toe_hit_ground
        done = jnp.logical_or(fallen, max_steps_reached)
        return done
    
    def _add_perturbations(
            self, rng: jax.Array, qpos: jax.Array, qvel: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Add uniform random perturbations to base and motor positions/velocities."""
        # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), z=+U(0, 0.2), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)

        rng, key = jax.random.split(rng)
        dz = jax.random.uniform(key, (1,), minval=0.0, maxval=0.2)

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math_utils.euler2quat(jnp.array([0.0, 0.0, yaw[0]]))

        
        qpos = qpos.at[QPosIdx.BASE_XY].add(dxy)
        qpos = qpos.at[QPosIdx.BASE_HEIGHT].add(dz)
        qpos = qpos.at[QPosIdx.BASE_QUAT].set(quat)

        # qpos[MOTORS]=*U(0.8, 1.2)
        rng, key = jax.random.split(rng)
        qpos = qpos.at[QPosIdx.MOTORS].set(
            qpos[QPosIdx.MOTORS] * jax.random.uniform(key, (10,), minval=0.8, maxval=1.2)
        )

        # d(xyzrpy)=U(-0.5, 0.5)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[QVelIdx.BASE].add(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        )

        return rng, qpos, qvel
    
    def _randomize_pd_gain(self, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Returns PD gains with an uncertainty applied."""
        rng, key = jax.random.split(rng)
        p_gain = (self._config.p_gain *
                  jax.random.uniform(
                      key,
                      shape=self._config.p_gain.shape,
                      minval=1.0 - self._config.pd_uncertainty,
                      maxval=1.0 + self._config.pd_uncertainty
                  )
        )
        rng, key = jax.random.split(rng)
        d_gain = (self._config.d_gain *
                  jax.random.uniform(
                      key,
                      shape=self._config.d_gain.shape,
                      minval=1.0 - self._config.pd_uncertainty,
                      maxval=1.0 + self._config.pd_uncertainty
                  )
        )
        return rng, p_gain, d_gain

    def _action_norm2actual(self, action: jax.Array) -> jax.Array:
        """Scales normalized actions [-1, 1] to actual motor joint ranges."""
        return self._jnt_soft_lowers + (action + 1) / 2.0 * (
            self._jnt_soft_uppers - self._jnt_soft_lowers
        )
    
    def _pd_control(
            self,
            data: mjx.Data,
            pos_targets: jax.Array,
            p_gain: jax.Array,
            d_gain: jax.Array,
            torque_lb: jax.Array,
            torque_ub: jax.Array,
    ) -> jax.Array:
        """Computes PD control torques for the actuated joints."""
        # Current motor positions and velocities
        motor_qpos = data.qpos[QPosIdx.MOTORS]
        motor_qvel = data.qvel[QVelIdx.MOTORS]

        # PD control
        pos_err = pos_targets - motor_qpos
        vel_err = -motor_qvel  # Target vel is zero

        # jax.debug.print("pos_err: {}", pos_err)
        # jax.debug.print("vel_err: {}", vel_err)

        torques = p_gain * pos_err + d_gain * vel_err

        return jnp.clip(torques, torque_lb, torque_ub)
    
    def _has_fallen(self, data: mjx.Data) -> jax.Array:
        """Returns True if Cassie has fallen (pelvis height below threshold)."""
        pelvis_z = data.qpos[QPosIdx.BASE_HEIGHT].squeeze()
        return pelvis_z < FALLING_THRESHOLD
