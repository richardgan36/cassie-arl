from typing import Any, Dict, Optional, Union
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax
import mujoco as mj
import mujoco.mjx as mjx
from flax import struct
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

import cassie_arl.rl_env.math_utils as math_utils
from cassie_arl.config.cassie_consts import (
    FOOT_OFFSET,
    TARSUS_HIT_GROUND_THRESHOLD,
    FALLING_THRESHOLD,
    FOOT_CONTACT_THRESHOLD,
    QPosIdx,
    QVelIdx,
    JntRangeIdx
)

script_dir = Path(__file__).parent
CASSIE_SCENE_XML = script_dir.parent / "models" / "scene.xml"


@struct.dataclass
class RewardComponents:
    """PyTree for per-step reward components kept in state.info.

    Storing rewards in a dataclass makes the structure static and JIT-friendly
    (vs an arbitrary dict). All leaves are jax.Arrays with scalar shape ().
    """
    alive: jax.Array
    pelvis_height: jax.Array
    pelvis_lin_vel: jax.Array
    pelvis_ang_vel: jax.Array
    pelvis_tilt: jax.Array
    motor_ref_error: jax.Array
    action_rate: jax.Array
    torques: jax.Array
    gain_rate: jax.Array

    @classmethod
    def zeros(cls) -> "RewardComponents":
        z = jnp.zeros(())
        return cls(
            alive=z,
            pelvis_height=z,
            pelvis_lin_vel=z,
            pelvis_ang_vel=z,
            pelvis_tilt=z,
            motor_ref_error=z,
            action_rate=z,
            torques=z,
            gain_rate=z,
        )


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        # --------------------------------
        # Required simulation parameters
        # --------------------------------
        ctrl_dt=0.01,  # 100 Hz
        sim_dt=0.002,  # Match "timestep" in MJCF
        episode_length=1000,  # 10 seconds at ctrl_dt=0.01

        # -------------------
        # Custom parameters
        # -------------------

        # Number of previous steps to include in the observation history.
        # Total stacked timesteps in obs = history_len + 1 (t plus history)
        history_len=2,

        # --- PD control parameters ---
        max_p_gain=8.0,  # Maximum Kp that can be learned by the agent
        max_d_gain=0.8,  # Maximum Kd that can be learned by the agent

        # max_joint_delta_frac is the fraction of the total joint range
        # that the action can command as a delta from the standing pose.
        max_joint_delta_frac=0.7,

        # --- Reset noise configuration ---
        reset_noise_config=config_dict.create(
            level=1.2,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                xy=jnp.array([-0.1, 0.1]),            # Additive
                z=jnp.array([0, 0.05]),               # Additive
                yaw=jnp.array([-3.14, 3.14]),         # Additive
                roll_pitch=jnp.array([-0.05, 0.05]),  # Additive
                motors=jnp.array([-0.03, 0.03]),      # Multiplicative: Motors *= U(1-0.03, 1+0.03)
                dxyz=jnp.array([-0.1, 0.1]),          # Additive
                drpy=jnp.array([-0.12, 0.12]),        # Additive
            ),
        ),

        # --- Reward function configuration ---
        reward_config=config_dict.create(
            weights=config_dict.create(
                alive=1.5,  # Initially 2.0 but reduced since other costs have been reduced
                pelvis_height=-0.0,  # Initially -0.3, but after training, the agent discovers it can get high reward without tracking height closely
                pelvis_lin_vel=-0.5,  # Initially -0.3 but increased to encourage stability
                pelvis_ang_vel=-0.6,  # Initially -0.4 but increased to encourage stability
                pelvis_tilt=-0.2,  # The standing pose found by the agent has a slight tilt, so this cost is reduced to avoid penalizing that too much
                motor_ref_error=-0.0,  # Initially -0.8 but removed because the agent learnt a standing pose that is different from the reference pose
                action_rate=-0.2,
                torques=-0.05,
                gain_rate=-0.0,  # Initially -0.1 to encourate constant gains but removed now that the agent has already learned this behavior 
            ),
        ),
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

        standing_quat = self._init_qpos[QPosIdx.BASE_QUAT]
        self._standing_base_rpy = math_utils.quat2euler(standing_quat)

        jnt_lowers = self._mj_model.jnt_range[JntRangeIdx.MOTORS, 0]
        jnt_uppers = self._mj_model.jnt_range[JntRangeIdx.MOTORS, 1]
        jnt_ranges = jnt_uppers - jnt_lowers
        self._max_jnt_deltas = jnp.array(jnt_ranges * self._config.max_joint_delta_frac)

        self._torque_lowers = jnp.array(self._mj_model.actuator_ctrlrange[:, 0])
        self._torque_uppers = jnp.array(self._mj_model.actuator_ctrlrange[:, 1])

        # -------------------
        # Precomputed constants for rewards/costs (avoid repeated pow/div)
        # -------------------
        # Reference pelvis height
        self._pelvis_height_ref = 0.985

        # Inverse-squared scales (1 / scale^2) for various costs
        self._inv_height_err_scale_sq = jnp.array(1.0 / (0.02 ** 2))
        self._inv_pelvis_lin_vel_scale = jnp.array(1.0 / (0.2 ** 2))
        self._inv_pelvis_ang_vel_scale = jnp.array(1.0 / (0.15 ** 2))
        self._inv_tilt_err_scale_sq = jnp.array(1.0 / (0.17 ** 2))
        self._inv_motor_ref_err_scale_sq = jnp.array(1.0 / (0.08 ** 2))

        # Action-rate normalization per joint: "full allowed delta traversed in ~0.25s"
        # Per-step scale = max_delta / (0.25 / dt) = max_delta / steps_per_quarter_sec
        steps_per_quarter_sec = jnp.maximum(0.25 / self.dt, 1.0)
        per_step_target_scale = self._max_jnt_deltas / steps_per_quarter_sec  # (10,)
        # Guard against tiny scales to avoid blow-ups
        per_step_target_scale = jnp.maximum(per_step_target_scale, 1e-6)
        self._inv_action_rate_scales_sq = 1.0 / (per_step_target_scale ** 2)  # (10,)

        # Torque scale is per-actuator; use upper bounds (assumed symmetric)
        # cost ~ mean((tau / (ub/2))^2) == mean((tau^2) * (2/ub)^2)
        self._inv_torque_scales_sq = (2.0 / self._torque_uppers) ** 2

        # Gain-rate scales (guarded against 0)
        self._p_gain_rate_scale = jnp.maximum(self._config.max_p_gain / 25.0, 1e-6)
        self._d_gain_rate_scale = jnp.maximum(self._config.max_d_gain / 25.0, 1e-6)
        self._inv_p_gain_rate_scale_sq = jnp.array(1.0 / (self._p_gain_rate_scale ** 2))
        self._inv_d_gain_rate_scale_sq = jnp.array(1.0 / (self._d_gain_rate_scale ** 2))

        # def geoms_of_body(model, body_id):
        #     start = model.body_geomadr[body_id]
        #     count = model.body_geomnum[body_id]
        #     geom_ids = jnp.arange(start, start + count)
        #     return geom_ids

        # MJCF geom and body IDs
        # self._floor_gid = self._mj_model.geom("floor").id
        self._pelvis_id = self._mj_model.body("cassie-pelvis").id
        self._left_foot_id = self._mj_model.body("left-foot").id
        self._right_foot_id = self._mj_model.body("right-foot").id
        self._left_tarsus_id = self._mj_model.body("left-tarsus").id
        self._right_tarsus_id = self._mj_model.body("right-tarsus").id

        # self._left_foot_gid = geoms_of_body(self._mj_model, self._left_foot_id)
        # self._right_foot_gid = geoms_of_body(self._mj_model, self._right_foot_id)

        # self._push_target_body_id = self._mj_model.body(self._config.pushes.target_body).id
    
    # ----------------------------------------------------------------------
    # Required abstract methods/properties
    # ----------------------------------------------------------------------

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        """Number of action dimensions.
        
        Size is 16. The first 10 actions correspond to joint angle deltas from
        the standing pose. The next 3 actions are the proportional gain (Kp)
        for [hip roll/yaw], [hip pitch / knee], [foot] respectively.
        The last 3 actions are the derivative gain (Kd) for the same groups.
        """
        return self._mjx_model.nu + 6

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

        data = mjx_env.init(self._mjx_model, qpos=qpos, qvel=qvel)
        # Use zeros torques with correct actuator dimension for initial obs
        zero_torques = jnp.zeros((self._mjx_model.nu,))
        obs_single = self._get_obs(data, zero_torques)

        # Initialize history buffer with the initial observation repeated
        hist_len = int(self._config.history_len) + 1  # t plus previous steps
        obs_history = jnp.tile(obs_single[None, :], (hist_len, 1))
        obs = obs_history.reshape(-1)

        info = {
            "rng": rng,
            "step": 0,
            "reward_components": RewardComponents.zeros(),
            "pos_targets": jnp.zeros((self._mjx_model.nu,)),
            # Store last per-joint PD gains to compute gain-rate cost
            "last_p_gains": jnp.zeros((self._mjx_model.nu,)),
            "last_d_gains": jnp.zeros((self._mjx_model.nu,)),
            # Rolling observation history buffer of shape (history_len+1, obs_dim)
            "obs_history": obs_history,
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

        # Action consists of joint deltas + PD gains
        nu = self._mjx_model.nu
        jnt_deltas_raw = action[:nu]
        p_gains_raw = action[nu:nu+3]  # Kp for [hip roll/yaw], [hip pitch / knee], [foot]
        d_gains_raw = action[nu+3:]    # Kd for same groups

        p_gains_raw_scaled = (p_gains_raw + 1.0) / 2.0 * self._config.max_p_gain  # Ensure positive
        d_gains_raw_scaled = (d_gains_raw + 1.0) / 2.0 * self._config.max_d_gain  # Ensure positive
        # Expand grouped gains to per-joint gains (10 actuated joints):
        # per leg order = [hip roll, hip yaw, hip pitch, knee, foot]
        per_leg_p = jnp.array([
            p_gains_raw_scaled[0],  # hip roll
            p_gains_raw_scaled[0],  # hip yaw
            p_gains_raw_scaled[1],  # hip pitch
            p_gains_raw_scaled[1],  # knee
            p_gains_raw_scaled[2],  # foot
        ])
        per_leg_d = jnp.array([
            d_gains_raw_scaled[0],
            d_gains_raw_scaled[0],
            d_gains_raw_scaled[1],
            d_gains_raw_scaled[1],
            d_gains_raw_scaled[2],
        ])
        p_gains = jnp.tile(per_leg_p, 2)  # (10,)
        d_gains = jnp.tile(per_leg_d, 2)  # (10,)

        pos_targets = self._action_to_jnt_targets(jnt_deltas_raw)  # The first 10 actions are joint angle deltas in [-1, 1]

        # Run PD control at simulator substep frequency (sim_dt):
        # Hold pos_targets and gains constant within this control interval,
        # but recompute torques each substep using the latest q, qdot.
        def _pd_substep(_: int, carry):
            data_carry, _last_tau = carry
            tau = self._pd_control(data_carry, pos_targets, p_gains, d_gains)
            tau = jnp.clip(tau, self._torque_lowers, self._torque_uppers)
            data_next = mjx_env.step(self._mjx_model, data_carry, tau, 1)
            return (data_next, tau)

        data, torques = lax.fori_loop(
            0,
            self.n_substeps,
            _pd_substep,
            (state.data, jnp.zeros((self._mjx_model.nu,), dtype=jnp.float32)),  # Initial torque value not used but needed for carry
        )

        # Build single-step obs, then update the rolling history buffer
        obs_single = self._get_obs(data, torques)
        hist = state.info["obs_history"]
        # Prepend newest obs and drop the oldest for most-recent-first ordering
        new_hist = hist.at[1:].set(hist[:-1])
        new_hist = new_hist.at[0].set(obs_single)
        obs = new_hist.reshape(-1)

        per_step_raw = self._get_reward(
                            data,
                            state.info,
                            pos_targets,
                            torques,
                            p_gains,
                            d_gains,
                        )

        # Scale each component by its configured weight
        per_step_weighted = {k: per_step_raw[k] * self._config.reward_config.weights[k] for k in per_step_raw}

        # Per-step components are integrated over dt
        reward = jnp.sum(jnp.array(list(per_step_weighted.values()))) * self.dt

        new_step = state.info.get("step", 0) + 1
        
        done = self._get_termination(data, jnp.array(new_step))
        done = jnp.array(done, dtype=reward.dtype)

        new_info = {
            **state.info,
            "step": new_step,
            "rng": rng,
            "reward_components": RewardComponents(**per_step_weighted),
            "pos_targets": pos_targets,
            "last_p_gains": p_gains,
            "last_d_gains": d_gains,
            "obs_history": new_hist,
        }
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            info=new_info
        )

    def _get_obs(self, data: mjx.Data, last_torques: jax.Array) -> jax.Array:
        """Constructs observation from mjx.Data (Cassie)."""
        # TODO: separate "state" and "privileged state"
        # TODO: We are now using joint angle deltas. Consider changing the action
        #       to represent deltas relative to standing pose as well.
        # TODO: consider adding foot forces
        # TODO: consider adding dual history architecture (Z Li). Probably not all
        #       that useful right now because state is relatively Markovian. But may
        #       be useful later when noise and external pushes are added.

        # Base orientation (world->body quaternion in MuJoCo convention)
        base_quat = data.qpos[QPosIdx.BASE_QUAT]

        # Yaw-invariant (tilt-only) quaternion: remove yaw component
        rpy = math_utils.quat2euler(base_quat)  # [roll, pitch, yaw]
        tilt_quat = math_utils.euler2quat(jnp.array([rpy[0], rpy[1], 0.0]))

        pelvis_height = data.qpos[QPosIdx.BASE_HEIGHT]

        # Joint positions / velocities
        motor_qpos = data.qpos[QPosIdx.MOTORS]
        motor_qvel = data.qvel[QVelIdx.MOTORS]

        # World-frame linear & angular velocities
        lin_vel_world = data.qvel[QVelIdx.BASE_LIN_VEL]
        ang_vel_body = data.qvel[QVelIdx.BASE_ANG_VEL]  # Angular velocity is already in body frame

        # Rotate world velocities into body frame
        lin_vel_body = math_utils.vec_world_to_body(base_quat, lin_vel_world)
        # Use difference between current motor angles and the standing pose
        motor_qpos_delta = motor_qpos - self._standing_jnt_angles

        # Foot contacts (approximate, based on foot height)
        left_foot_contact = self._left_foot_height(data) < FOOT_CONTACT_THRESHOLD
        right_foot_contact = self._right_foot_height(data) < FOOT_CONTACT_THRESHOLD
        feet_contact = jnp.array([left_foot_contact, right_foot_contact], dtype=jnp.float32)

        obs = jnp.concatenate([
            pelvis_height,
            tilt_quat,
            motor_qpos_delta,
            lin_vel_body,
            ang_vel_body,
            motor_qvel,
            feet_contact,
            last_torques
        ])

        return obs

    def _get_reward(
            self,
            data: mjx.Data,
            info: Dict[str, Any],
            pos_targets: jax.Array,
            torques: jax.Array,
            p_gains: jax.Array,
            d_gains: jax.Array,
        ) -> Dict[str, jax.Array]:
        """
        Computes reward components.

        All rewards/costs are in [0, 1]. Their weights in self._config.reward_config.weights
        determine their relative importance and sign.
        """

        gain_rate_cost = lax.cond(
            jnp.array(info.get("step", 0)) == 0,
            lambda _: jnp.array(0.0),
            lambda _: self._cost_gain_rate(
                p_gains,
                d_gains,
                info.get("last_p_gains", jnp.zeros_like(p_gains)),
                info.get("last_d_gains", jnp.zeros_like(d_gains)),
            ),
            operand=None,
        )

        per_step_rewards = {
            "alive": self._reward_alive(),
            "pelvis_height": self._cost_pelvis_height(data),
            "pelvis_lin_vel": self._cost_pelvis_lin_vel(data),
            "pelvis_ang_vel": self._cost_pelvis_ang_vel(data),
            "pelvis_tilt": self._cost_pelvis_tilt(data),
            "motor_ref_error": self._cost_motor_reference_error(data),
            "action_rate": self._cost_action_rate(pos_targets, info["pos_targets"]),
            "torques": self._cost_torques(torques),
            "gain_rate": gain_rate_cost,
        }

        return per_step_rewards

    def _reward_alive(self) -> jax.Array:
        """Reward for staying 'alive' (not falling over)."""
        return jnp.array(1.0)
    
    def _cost_pelvis_height(self, data: mjx.Data) -> jax.Array:
        """Cost for pelvis height deviation from standing height."""
        pelvis_height = data.qpos[QPosIdx.BASE_HEIGHT]  # Shape (1,)
        # Use reference standing height computed in _post_init
        height_err = pelvis_height - self._pelvis_height_ref
        # (err / s)^2 == err^2 * (1/s^2)
        cost = (height_err[0] ** 2) * self._inv_height_err_scale_sq
        return jnp.clip(cost, 0.0, 1.0)
    
    def _cost_pelvis_lin_vel(self, data: mjx.Data) -> jax.Array:
        """Cost for pelvis linear velocity."""
        pelvis_lin_vel = data.qvel[QVelIdx.BASE_LIN_VEL]
        v_sq = jnp.mean(pelvis_lin_vel**2)
        cost = v_sq * self._inv_pelvis_lin_vel_scale
        return jnp.clip(cost, 0.0, 1.0)
    
    def _cost_pelvis_ang_vel(self, data: mjx.Data) -> jax.Array:
        ang_vel = data.qvel[QVelIdx.BASE_ANG_VEL]
        ang_vel_sq_mean = jnp.mean(ang_vel**2)
        cost = ang_vel_sq_mean * self._inv_pelvis_ang_vel_scale
        return jnp.clip(cost, 0.0, 1.0)

    def _cost_pelvis_tilt(self, data: mjx.Data) -> jax.Array:
        """Cost for pelvis orientation (deviation from standing)."""
        # Get base quaternion
        base_quat = data.qpos[QPosIdx.BASE_QUAT]
        rpy = math_utils.quat2euler(base_quat)

        # Only roll and pitch
        orientation_err = math_utils.angle_diff(rpy[:2], self._standing_base_rpy[:2])

        # Mean squared error, normalized
        # (err / s)^2 == err^2 * (1/s^2)
        orientation_cost = jnp.mean((orientation_err ** 2)) * self._inv_tilt_err_scale_sq

        # Clip to [0,1]
        return jnp.clip(orientation_cost, 0.0, 1.0)

    def _cost_motor_reference_error(self, data: mjx.Data) -> jax.Array:
        """Cost for deviation of the motor angles from reference standing pose."""
        motor_qpos = data.qpos[QPosIdx.MOTORS]
        err = motor_qpos - self._standing_jnt_angles
        cost = jnp.mean((err ** 2)) * self._inv_motor_ref_err_scale_sq
        return jnp.clip(cost, 0.0, 1.0)

    def _cost_action_rate(self, pos_targets: jax.Array, last_pos_targets: jax.Array) -> jax.Array:
        """Cost for large changes in action between steps.
        
        The 'action' here is interpreted as the target joint angles.
        """
        # Normalize per joint by the "allowed" per-step change and average
        act_rate = pos_targets - last_pos_targets  # (10,)
        cost = jnp.mean((act_rate ** 2) * self._inv_action_rate_scales_sq)
        return jnp.clip(cost, 0.0, 1.0)
    
    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        """Cost for large torques."""
        # Incur max cost if using half of max torque
        cost = jnp.mean((torques ** 2) * self._inv_torque_scales_sq)
        return jnp.clip(cost, 0.0, 1.0)

    def _cost_gain_rate(
            self,
            p_gains: jax.Array,
            d_gains: jax.Array,
            last_p_gains: jax.Array,
            last_d_gains: jax.Array,
        ) -> jax.Array:
        """Cost for rapid per-step changes in learned PD gains.

        Normalization: a change of ~max_gain over about 0.25s (i.e., 25 steps at dt=0.01)
        should yield near-max cost. Therefore, per-step scale ~= max_gain / 25.
        """
        dp = p_gains - last_p_gains
        dd = d_gains - last_d_gains

        # Mean squared normalized changes for both P and D
        cost_p = jnp.mean((dp ** 2)) * self._inv_p_gain_rate_scale_sq
        cost_d = jnp.mean((dd ** 2)) * self._inv_d_gain_rate_scale_sq
        cost = 0.5 * (cost_p + cost_d)
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

    def _action_to_jnt_targets(self, joint_deltas_normalized: jax.Array) -> jax.Array:
        """Scales normalized actions [-1, 1] to motor joint angles.

        The action is interpreted as normalized deltas of the 10 actuated
        joints from the standing pose angles. The scaling is linear:
            delta = action * max_delta
            target = standing + delta
        
        Note:
            The joint targets returned by this function do NOT respect
            the joint limits, and this is intentional - in order to
            maintain the maximum allowed joint angle, the PD controller
            may require the target to be outside the joint limits.
        """
        return self._standing_jnt_angles + joint_deltas_normalized * self._max_jnt_deltas

    # ------------------------------------------------------------------
    # Random push helpers (modular)
    # ------------------------------------------------------------------

    def _push_steps_from_seconds(self, seconds: float) -> jax.Array:
        """Convert seconds to an integer number of control steps as a JAX int32 scalar.

        Ensures 0 when seconds <= 0, otherwise at least 1.
        """
        sec = jnp.array(seconds, dtype=jnp.float32)
        steps_f = jnp.round(sec / self.dt)
        steps_i = jnp.maximum(steps_f.astype(jnp.int32), jnp.array(1, dtype=jnp.int32))
        steps_i = jnp.where(sec <= 0.0, jnp.array(0, dtype=jnp.int32), steps_i)
        return steps_i

    def _push_start_prob_per_step(self, start_rate_hz: float) -> jax.Array:
        """Poisson-process start probability per control step for a given rate in Hz.

        p = 1 - exp(-lambda * dt), invariant to discretization.
        """
        lam_dt = jnp.maximum(jnp.array(start_rate_hz, dtype=jnp.float32) * self.dt, 0.0)
        return (1.0 - jnp.exp(-lam_dt))

    def _sample_push_force6(self, rng: jax.Array, push_cfg) -> tuple[jax.Array, jax.Array]:
        """Sample a 6D wrench [fx, fy, fz, tx, ty, tz] in world frame given push config."""
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        if getattr(push_cfg, "direction_mode", "horizontal") == "horizontal":
            theta = jax.random.uniform(rng1, (), minval=0.0, maxval=2.0 * jnp.pi)
            direction = jnp.array([jnp.cos(theta), jnp.sin(theta), 0.0])
        else:
            # Any-3D unit vector (sample from normal and normalize)
            v = jax.random.normal(rng1, (3,))
            norm = jnp.maximum(jnp.linalg.norm(v), 1e-6)
            direction = v / norm

        f_mag = jax.random.uniform(rng2, (), minval=push_cfg.force_range[0], maxval=push_cfg.force_range[1])
        t_mag = jax.random.uniform(rng3, (), minval=push_cfg.torque_range[0], maxval=push_cfg.torque_range[1])
        force = f_mag * direction
        torque = jnp.array([0.0, 0.0, t_mag])  # By default apply torque about z
        return rng3, jnp.concatenate([force, torque])

    def _update_push_state(
        self,
        rng: jax.Array,
        push_state,
        push_cfg: Optional[config_dict.ConfigDict],
        push_enabled: bool,
    ) -> tuple[jax.Array, jax.Array]:
        """Advance push state by one control step and return wrench to apply.

        Returns (rng, new_push_state, force6_to_apply)
        """
        if not push_enabled:
            # No pushes; decay any ongoing state to idle
            zero = jnp.zeros((6,), dtype=jnp.float32)
            return rng, zero

        # Convert invariant config into per-step quantities
        start_p = self._push_start_prob_per_step(push_cfg.start_rate_hz)
        min_interval_steps = self._push_steps_from_seconds(push_cfg.min_interval_s)
        duration_steps = self._push_steps_from_seconds(push_cfg.duration_s)

        steps_rem = push_state.steps_remaining
        cooldown = push_state.cooldown
        prev_force6 = push_state.force6

        # Decide if we start a push this step
        rng, key_start = jax.random.split(rng)
        start_prob = jax.random.uniform(key_start, ())
        can_start = jnp.logical_and(steps_rem == 0, cooldown == 0)
        start_now = jnp.logical_and(can_start, start_prob < start_p)

        # Sample new force if starting; otherwise keep previous
        rng, new_force6 = lax.cond(
            start_now,
            lambda r: self._sample_push_force6(r, push_cfg),
            lambda r: (r, prev_force6),
            operand=rng,
        )

        # Update timers
        new_steps = lax.select(start_now, duration_steps.astype(jnp.int32), steps_rem)
        active = new_steps > 0
        steps_after = jnp.where(active, new_steps - 1, new_steps)

        idle = jnp.logical_not(active)
        cooldown_dec = jnp.maximum(cooldown - 1, 0)
        new_cooldown = lax.select(
            start_now,
            min_interval_steps.astype(jnp.int32),
            jnp.where(idle, cooldown_dec, cooldown),
        )

        force6_to_apply = jnp.where(active, new_force6, jnp.zeros_like(new_force6))

        return rng, force6_to_apply

    def _apply_external_force(self, data: mjx.Data, force6: jax.Array, substeps: int, tau: jax.Array) -> mjx.Data:
        """Apply a 6D wrench to the configured body for the given number of substeps and step the sim."""
        # Build xfrc_applied array
        xfrc = jnp.zeros((self._mjx_model.nbody, 6), dtype=data.xfrc_applied.dtype)
        xfrc = xfrc.at[self._push_target_body_id].set(force6)

        def step_once(_: int, d: mjx.Data):
            return mjx_env.step(self._mjx_model, d.replace(xfrc_applied=xfrc), tau, 1)

        return lax.fori_loop(0, substeps, step_once, data)

    def _pd_control(
            self,
            data: mjx.Data,
            pos_targets: jax.Array,
            p_gain: jax.Array,
            d_gain: jax.Array,
    ) -> jax.Array:
        """Computes PD control torques."""
        motor_qpos = data.qpos[QPosIdx.MOTORS]
        raw_motor_qvel = data.qvel[QVelIdx.MOTORS]
        pos_error = pos_targets - motor_qpos
        vel_error = -raw_motor_qvel  # Desire zero velocity
        torques = p_gain * pos_error + d_gain * vel_error
        return torques

    def _add_perturbations(
            self, rng: jax.Array, qpos: jax.Array, qvel: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Add uniform random perturbations to base and motor positions/velocities."""
        noise_level = self._config.reset_noise_config.level
        rng, key = jax.random.split(rng)
        qpos = qpos.at[QPosIdx.BASE_XY].add(
            jax.random.uniform(
                key,
                (2,),
                minval=self._config.reset_noise_config.scales.xy[0] * noise_level,
                maxval=self._config.reset_noise_config.scales.xy[1] * noise_level
            )
        )

        rng, key = jax.random.split(rng)
        qpos = qpos.at[QPosIdx.BASE_HEIGHT].add(
            jax.random.uniform(
                key,
                (1,),
                minval=self._config.reset_noise_config.scales.z[0] * noise_level,
                maxval=self._config.reset_noise_config.scales.z[1] * noise_level
            )
        )

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(
            key,
            (1,),
            minval=self._config.reset_noise_config.scales.yaw[0] * noise_level,
            maxval=self._config.reset_noise_config.scales.yaw[1] * noise_level
        )
        rng, key = jax.random.split(rng)
        roll_pitch = jax.random.uniform(
            key,
            (2,),
            minval=self._config.reset_noise_config.scales.roll_pitch[0] * noise_level,
            maxval=self._config.reset_noise_config.scales.roll_pitch[1] * noise_level
        )
        quat = math_utils.euler2quat(jnp.concatenate([roll_pitch, yaw]))
        # Compose: new orientation = delta * base
        new_quat = math_utils.quat_mul(quat, qpos[QPosIdx.BASE_QUAT])
        new_quat = new_quat / jnp.linalg.norm(new_quat)  # Normalize
        qpos = qpos.at[QPosIdx.BASE_QUAT].set(new_quat)

        # Noise is multiplicative on motor angles
        rng, key = jax.random.split(rng)
        qpos = qpos.at[QPosIdx.MOTORS].multiply(
            jax.random.uniform(
                key,
                (10,),
                minval=1 + self._config.reset_noise_config.scales.motors[0] * noise_level,
                maxval=1 + self._config.reset_noise_config.scales.motors[1] * noise_level
            )
        )

        rng, key = jax.random.split(rng)
        dxyz_delta = jax.random.uniform(
            key, 
            (3,),
            minval=self._config.reset_noise_config.scales.dxyz[0] * noise_level,
            maxval=self._config.reset_noise_config.scales.dxyz[1] * noise_level
        )
        rng, key = jax.random.split(rng)
        drpy_delta = jax.random.uniform(
            key,
            (3,),
            minval=self._config.reset_noise_config.scales.drpy[0] * noise_level,
            maxval=self._config.reset_noise_config.scales.drpy[1] * noise_level
        )

        qvel = qvel.at[QVelIdx.BASE].add(
            jnp.concatenate([dxyz_delta, drpy_delta])
        )

        return rng, qpos, qvel

    def _left_foot_height(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self._left_foot_id, 2] - FOOT_OFFSET

    def _right_foot_height(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self._right_foot_id, 2] - FOOT_OFFSET



