from typing import Any, Dict, Optional, Union
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

from cassie_arl.config.cassie_consts import *
import cassie_arl.rl_env.math_utils as math_utils


script_dir = Path(__file__).parent
CASSIE_SCENE_XML = script_dir / ".." / "models" / "scene.xml"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                hip_pos=0.03,  # rad
                kfe_pos=0.05,
                ffe_pos=0.08,
                faa_pos=0.03,
                joint_vel=1.5,  # rad/s
                gravity=0.05,
                linvel=0.1,
                gyro=0.2,  # angvel.
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                alive=1.0,
                pelvis_lin_vel=-0.7,
                pelvis_orientation=-0.5,
                motor_ref_error=-0.6
            ),
        ),
        lin_vel_x=[-1.0, 1.0],
        lin_vel_y=[-1.0, 1.0],
        ang_vel_yaw=[-1.0, 1.0],
    )


class CassieEnv(mjx_env.MjxEnv):
    """Cassie environment built on MJX, compatible with Brax PPO."""

    def __init__(
            self,
            xml_path: str = CASSIE_SCENE_XML.as_posix(),
            config: config_dict.ConfigDict = default_config(),
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None
    ):
        super().__init__(config, config_overrides)
        # Load MuJoCo model and MJX version
        self._xml_path = xml_path
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mjx_model = mjx.put_model(self._mj_model)

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._post_init()

    def _post_init(self):
        self._init_q = jnp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = jnp.array(self._mj_model.keyframe("home").qpos[7:])

        # Note: First joint is freejoint.
        self._jnt_lowers, self._jnt_uppers = self.mj_model.jnt_range[1:].T
        jnt_c = (self._jnt_lowers + self._jnt_uppers) / 2
        jnt_r = self._jnt_uppers - self._jnt_lowers
        self._jnt_soft_lowers = jnt_c - 0.5 * jnt_r * self._config.soft_joint_pos_limit_factor
        self._jnt_soft_uppers = jnt_c + 0.5 * jnt_r * self._config.soft_joint_pos_limit_factor

        act_ctrl = jnp.array(self._mj_model.actuator_ctrlrange.T)  # shape (2, nu)
        act_ctrl = act_ctrl.at[:, [LEFT_HIP_ROLL_IDX, RIGHT_HIP_ROLL_IDX]].multiply(0.5)
        act_ctrl = act_ctrl.at[:, [LEFT_HIP_YAW_IDX, RIGHT_HIP_YAW_IDX]].multiply(0.8)
        self._actuator_soft_bounds = act_ctrl

        self._pelvis_id = self._mj_model.body("cassie-pelvis").id

        # hip_indices = []
        # hip_joint_names = ["HR", "HAA"]
        # for side in ["LL", "LR"]:
        #     for joint_name in hip_joint_names:
        #         hip_indices.append(
        #             self._mj_model.joint(f"{side}_{joint_name}").qposadr - 7
        #         )
        # self._hip_indices = jnp.array(hip_indices)

        # knee_indices = []
        # for side in ["LL", "LR"]:
        #     knee_indices.append(self._mj_model.joint(f"{side}_KFE").qposadr - 7)
        # self._knee_indices = jnp.array(knee_indices)

        # # fmt: off
        # self._weights = jnp.array([
        #     1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # left leg.
        #     1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # right leg.
        # ])
        # # fmt: on

        # self._torso_body_id = self._mj_model.body(ROOT_BODY).id
        # self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        # self._site_id = self._mj_model.site("imu").id

        # self._feet_site_id = np.array(
        #     [self._mj_model.site(name).id for name in FEET_SITES]
        # )
        # self._floor_geom_id = self._mj_model.geom("floor").id
        # self._feet_geom_id = np.array(
        #     [self._mj_model.geom(name).id for name in FEET_GEOMS]
        # )

        # foot_linvel_sensor_adr = []
        # for site in FEET_SITES:
        #     sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
        #     sensor_adr = self._mj_model.sensor_adr[sensor_id]
        #     sensor_dim = self._mj_model.sensor_dim[sensor_id]
        #     foot_linvel_sensor_adr.append(
        #         list(range(sensor_adr, sensor_adr + sensor_dim))
        #     )
        # self._foot_linvel_sensor_adr = jnp.array(foot_linvel_sensor_adr)

        # qpos_noise_scale = np.zeros(12)
        # hip_ids = [0, 1, 2, 6, 7, 8]
        # kfe_ids = [3, 9]
        # ffe_ids = [4, 10]
        # faa_ids = [5, 11]
        # qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
        # qpos_noise_scale[kfe_ids] = self._config.noise_config.scales.kfe_pos
        # qpos_noise_scale[ffe_ids] = self._config.noise_config.scales.ffe_pos
        # qpos_noise_scale[faa_ids] = self._config.noise_config.scales.faa_pos
        # self._qpos_noise_scale = jnp.array(qpos_noise_scale)


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
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # ----------------------------------------------------------------------
    # Core env logic
    # ----------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resets Cassie to default pose + small random perturbations."""
        qpos = self._init_q
        qvel = jnp.zeros(self.mjx_model.nv)

        rng, key = jax.random.split(rng)

        # Example randomization: add small noise to root position/orientation
        qpos = qpos.at[0].add(jax.random.uniform(key, (), minval=-0.01, maxval=0.01))

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel)
        obs = self._get_obs(data)

        info = {
            "rng": rng,
            "step": 0,
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
        """Takes one control step in Cassie."""
        rng = state.info["rng"]
        rng, key = jax.random.split(rng)

        # Clip actions to [-1, 1] (scale to actuator range later if needed)
        action = jnp.clip(action, -1.0, 1.0)

        # Run physics for n_substeps
        actual_action = self._action_norm2actual(action)

        data = state.data
        data = mjx_env.step(
            self._mjx_model, state.data, actual_action, self.n_substeps
        )

        # New observation
        obs = self._get_obs(data)

        # Placeholder reward: keep pelvis upright
        rewards = self._get_reward(data, action)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        # reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        reward = sum(rewards.values()) * self.dt
        reward = jnp.clip(reward, -1000, 1000)

        # Termination
        new_step = state.info.get("step", 0) + 1
        
        done = self._get_termination(data, jnp.array(new_step))
        done = jnp.array(done, dtype=reward.dtype)

        new_info = {**state.info, "step": new_step, "rng": rng}

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
        # TODO: consider adding contact flags / foot heights / swing time / etc.
        # TODO: consider adding gravity vector relative to pelvis frame

        # qpos: joint + base positions
        # qvel: joint + base velocities
        # First 7 entries of qpos are free joint (base pos + quaternion)
        motor_qpos = data.qpos[MOTOR_IDX]   # exclude root pos/orientation
        motor_qvel = data.qvel[MOTOR_VEL_IDX]   # exclude root linear + angular vel
        pelvis_qvel = data.qvel[:6]

        # pelvis_lin_vel = data.qvel[:3]

        # Add pelvis orientation as a flat quaternion
        pelvis_quat = data.qpos[3:7]   # 4D unit quaternion

        # Pelvis height (z coord of root)
        pelvis_height = data.qpos[2:3] # shape (1,)

        # Concatenate into a single vector
        obs = jnp.concatenate([
            pelvis_quat, pelvis_height, motor_qpos, motor_qvel, pelvis_qvel
        ])

        return obs

    def _get_reward(self, data: mjx.Data, action: jax.Array) -> dict[str, jax.Array]:
        # TODO: reward for COM above support polygon
        # TODO: cost for large change in acceleration
        # TODO: cost for large torques
        # TODO: cost for deviation from standing pose
        return {
            "alive": self._reward_alive(data),
            "pelvis_lin_vel": self._cost_pelvis_lin_vel(data),
            "pelvis_orientation": self._cost_pelvis_orientation(data),
            "motor_ref_error": self._cost_motor_reference_error(data),
        }

    def _reward_alive(self, data: mjx.Data) -> jax.Array:
        """Reward for staying 'alive' (not falling over)."""
        return jnp.where(data.qpos[2] > 0.5, 1.0, 0.0)

    def _cost_pelvis_lin_vel(self, data: mjx.Data) -> jax.Array:
        """Cost for pelvis linear velocity."""
        pelvis_lin_vel = data.qvel[:3]
        v = jnp.sqrt(jnp.sum(pelvis_lin_vel**2))
        v_scale = 0.1  # Normalizing constant (m/s)
        cost = jnp.abs(v / v_scale)**2
        return jnp.clip(cost, 0, 1)

    def _cost_pelvis_orientation(self, data: mjx.Data) -> jax.Array:
        """Cost for pelvis orientation (deviation from standing)."""
        # Get pelvis quaternion
        pelvis_quat = data.qpos[3:7]
        rpy = math_utils.quat2euler(pelvis_quat)

        # Only roll and pitch
        orientation_err = math_utils.angle_diff(rpy[:2], STANDING_PELVIS_RPY[:2])

        # Mean squared error, normalized
        err_scale = 0.35  # radians
        orientation_cost = jnp.mean((orientation_err / err_scale) ** 2)

        # Clip to [0,1]
        return jnp.clip(orientation_cost, 0.0, 1.0)

    def _cost_motor_reference_error(self, data: mjx.Data) -> jax.Array:
        """Cost for deviation of the motor angles from reference standing pose."""
        motor_qpos = data.qpos[MOTOR_IDX]
        err = motor_qpos - MOTORS_STANDING_POSE
        err_scale = 0.35  # Normalizing constant (radians)
        cost = jnp.mean((err / err_scale)**2)
        return jnp.clip(cost, 0, 1)

    def _get_termination(self, data: mjx.Data, step: jax.Array) -> jax.Array:
        """Return True if Cassie has fallen or max timesteps reached."""
        # Pelvis height (z-coordinate)
        pelvis_z = data.qpos[2]
        fallen = pelvis_z < FALLING_THRESHOLD

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
    
    def _action_norm2actual(self, action: jax.Array) -> jax.Array:
        """Scales normalized actions [-1, 1] to actual actuator control range."""
        return self._actuator_soft_bounds[0] + (action + 1) / 2.0 * (
            self._actuator_soft_bounds[1] - self._actuator_soft_bounds[0]
        )
    
    def _set_motor_base_pos(model, data, motor_idx, base_idx, motor_pos, base_pos, iters=1500):
        def project_state(model, data, iters):
            # Abuses the MJX solver to get constrained forward kinematics.
            def body_fun(_, d):
                return mjx.step(model, d, ctrl=jnp.zeros(model.nu))
            return jax.lax.fori_loop(0, iters, body_fun, data)

        qpos = data.qpos.at[motor_idx].set(motor_pos)
        qpos = qpos.at[base_idx].set(base_pos)
        qvel = jnp.zeros_like(data.qvel)

        data = data.replace(qpos=qpos, qvel=qvel)
        data = project_state(model, data, iters=iters)
        return data
    