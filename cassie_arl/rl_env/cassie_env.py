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
                com_outside_support=-0.4,
                pelvis_lin_vel=-0.2,
                pelvis_tilt=-0.1,
                motor_ref_error=-0.2,
                airborne=-0.2,
                # Reward for lifting exactly one foot when COM is far from support
                lift_foot=0.3,
            ),
        ),
        # Parameters for the "lift foot" reward
        # If the COM is further than `com_outside_support_threshold` (m) from the
        # support polygon and exactly one foot is lifted with at least
        # `lift_foot_clearance` (m) clearance from the ground, the
        # environment returns a binary reward (1.0) for that component.
        com_outside_support_threshold=0.06,     # meters
        lift_foot_clearance=0.025,              # meters
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
        self._default_pose = jnp.array(self._mj_model.keyframe("home").qpos[7:])

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

        self._left_foot_gid = geoms_of_body(self._mj_model, self._left_foot_id)
        self._right_foot_gid = geoms_of_body(self._mj_model, self._right_foot_id)

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
                "com_outside_support": jnp.zeros(()),
                "airborne": jnp.zeros(()),
                "lift_foot": jnp.zeros(()),
            },
            "action": jnp.zeros((self.action_size,)), 
            # Whether the one-time "lift foot" reward has been granted
            "lift_foot_given": jnp.array(False),
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

        obs = self._get_obs(data)

        # Get reward components. `_get_reward` returns ((per_step_raw, event_raw), lift_given_new)
        (per_step_raw, event_raw), lift_given_new = self._get_reward(data, action, state.info)

        # Scale each component by its configured weight
        per_step_scaled = {k: per_step_raw[k] * self._config.reward_config.scales[k] for k in per_step_raw}
        event_scaled = {k: event_raw[k] * self._config.reward_config.scales[k] for k in event_raw}

        # Per-step components are integrated over dt; event components are added directly
        reward = sum(per_step_scaled.values()) * self.dt + sum(event_scaled.values())
        reward = jnp.clip(reward, -1000, 1000)

        new_step = state.info.get("step", 0) + 1
        
        done = self._get_termination(data, jnp.array(new_step))
        done = jnp.array(done, dtype=reward.dtype)

        new_info = {
            **state.info,
            "step": new_step,
            "rng": rng,
            "reward_components": {**per_step_scaled, **event_scaled},   # For debugging
            "action": action,                                           # For debugging
            "lift_foot_given": lift_given_new,
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

        motor_qpos = data.qpos[QPosIdx.MOTORS]
        motor_qvel = data.qvel[QVelIdx.MOTORS]
        pelvis_qvel = data.qvel[QVelIdx.BASE]

        # Add pelvis orientation as a flat quaternion
        pelvis_quat = data.qpos[QPosIdx.BASE_QUAT]
        gravity_in_pelvis_frame = math_utils.gravity_in_base_frame(pelvis_quat)

        # Pelvis height (z coord of root)
        pelvis_height = data.qpos[QPosIdx.BASE_HEIGHT]  # shape (1,)

        # Use difference between current motor angles and the standing pose
        motor_qpos_delta = motor_qpos - StandingPose.MOTOR_ANGLES

        # Foot contact info (left, right) as floats 0.0/1.0
        left_contact = self._is_in_contact_with_ground(data, self._left_foot_gid)
        right_contact = self._is_in_contact_with_ground(data, self._right_foot_gid)
        left_contact_f = jnp.where(left_contact, 1.0, 0.0)
        right_contact_f = jnp.where(right_contact, 1.0, 0.0)
        foot_contact = jnp.array([left_contact_f, right_contact_f])

        # Vector from COM to closest point on support polygon (XY)
        com_to_support = self._vector_com_to_support(data)

        # Concatenate into a single vector. Order chosen to keep base-state first,
        # then motor errors and velocities, then pelvis vel, then foot contact and com->support.
        obs = jnp.concatenate([
            pelvis_height,
            pelvis_quat,
            gravity_in_pelvis_frame,
            motor_qpos_delta,
            motor_qvel,
            pelvis_qvel,
            foot_contact,
            com_to_support,
        ])

        return obs

    def _get_reward(self, data: mjx.Data, action: jax.Array, info: Dict[str, Any]) -> tuple[tuple[dict[str, jax.Array], dict[str, jax.Array]], jax.Array]:
        """
        Computes reward components.

        All rewards/costs are in [0, 1]. Their weights in self._config.reward_config.scales
        determine their relative importance and sign.
        """
        # TODO: IMPORTANT: need to reward active recovery strategies, not just standing still
        # TODO: look into selective/adaptive rewards e.g. lift costs for movement when perturbing robot
        # TODO: cost for large change in acceleration
        # TODO: cost for large torques
        # TODO: reward for placing foot down depending on how close COM is to support polygon

        # Split components into per-step (integrated over dt) and event (one-time) rewards.
        per_step = {
            "alive": self._reward_alive(),
            "com_outside_support": self._cost_com_outside_support(data),
            "pelvis_lin_vel": self._cost_pelvis_lin_vel(data),
            "pelvis_tilt": self._cost_pelvis_tilt(data),
            "motor_ref_error": self._cost_motor_reference_error(data),
            "airborne": self._cost_airborne(data),
        }

        # Event-style components (one-time when triggered)
        event = {
            "fall": self._cost_fall(data),
        }

        # Compute lift_foot component and updated flag using event logic
        lift_comp, lift_given_new = self._reward_lift_foot_to_recover(data, info["lift_foot_given"])
        event["lift_foot"] = lift_comp

        return (per_step, event), lift_given_new

    def _reward_lift_foot_to_recover(
            self,
            data: mjx.Data,
            lift_foot_given: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
        """
        Reward 1.0 when COM is 'outside' the support polygon and
        exactly one foot is lifted by at least `lift_foot_clearance`.

        This encourages active recovery strategies (lifting one foot to reach
        or step) instead of keeping both feet planted and falling.
        Cassie is considered 'outside' the support polygon if the distance
        from the COM to the support polygon is greater than a threshold.

        Args:
            lift_foot_given: Boolean flag indicating whether the one-time
                reward has already been given for the current "COM outside"
                event. If True, no further reward is given until the COM
                returns inside the support polygon and then goes outside again.
        """
        # Distance vector from COM to support (2D)
        vec = self._vector_com_to_support(data)
        dist = jnp.linalg.norm(vec)

        # Check COM beyond threshold
        com_outside_support = dist >= self._config.com_outside_support_threshold

        # Foot clearances using body z positions (world frame)
        left_z = jnp.array(data.xpos[self._left_foot_id, 2]) - FOOT_OFFSET
        right_z = jnp.array(data.xpos[self._right_foot_id, 2]) - FOOT_OFFSET

        left_lifted = left_z >= self._config.lift_foot_clearance
        right_lifted = right_z >= self._config.lift_foot_clearance

        # Exactly one foot lifted
        exactly_one = jnp.logical_xor(left_lifted, right_lifted)

        # Should we give reward now? Event active (COM outside), not given yet for
        # this active period, and exactly one foot lifted.
        give_now = jnp.logical_and(jnp.logical_and(com_outside_support, jnp.logical_not(lift_foot_given)), exactly_one)

        # Raw component: if giving now, return 1.0 as a one-time event reward.
        cost = jnp.where(give_now, 1.0, 0.0)

        # Update flag: if COM is outside, keep previous or set flag if we just gave the reward.
        # If COM is inside, reset the flag so future outside-events can be rewarded
        lift_given_new = jnp.where(com_outside_support, jnp.logical_or(lift_foot_given, give_now), jnp.array(False))

        return cost, lift_given_new

    def _reward_alive(self) -> jax.Array:
        """Reward for staying 'alive' (not falling over)."""
        return jnp.array(1.0)
    
    def _cost_fall(self, data: mjx.Data) -> jax.Array:
        """One time cost for falling over."""
        fallen = self._has_fallen(data)
        # If fallen, return 1.0 as a one-time event cost. The caller will add
        return jnp.where(fallen, 1.0, 0.0)

    def _cost_com_outside_support(self, data: mjx.Data) -> jax.Array:
        """
        Cost for distance of the center of mass from the support polygon.

        The support polygon is a line segment if both feet are on the ground,
        or a point if one foot is on the ground. If both feet are in the air,
        the cost is zero. Cost for being airborne is penalized in _cost_airborne.
        """
        # Compute simplified distance to support (segment/point)
        vec_to_support = self._vector_com_to_support(data)
        dist = jnp.linalg.norm(vec_to_support)

        # Exponential scaling: reaches 0.4 at distance == sigma
        sigma = self._config.com_outside_support_threshold
        cost = 1 - jnp.exp(- (dist**2) / (2 * sigma**2))
        return jnp.clip(cost, 0.0, 1.0)
    
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
    
    def _cost_airborne(self, data: mjx.Data) -> jax.Array:
        """Cost for having both feet off the ground."""
        left_foot_contact = self._is_in_contact_with_ground(data, self._left_foot_gid)
        right_foot_contact = self._is_in_contact_with_ground(data, self._right_foot_gid)

        airborne = jnp.logical_and(jnp.logical_not(left_foot_contact), jnp.logical_not(right_foot_contact))
        return jnp.where(airborne, 1.0, 0.0)
    
    def _get_termination(self, data: mjx.Data, step: jax.Array) -> jax.Array:
        """Return True if Cassie has fallen or max timesteps reached."""
        fallen = self._has_fallen(data)

        max_steps = jnp.array(self._config.episode_length, dtype=step.dtype)
        max_steps_reached = step >= max_steps

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
        """
        Scales normalized actions [-1, 1] to actual motor joint angles.

        The action is interpreted as a normalized position target for each
        of the 10 actuated joints.
        """
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

    # ---------------- Support polygon & COM helpers ----------------
    def _is_in_contact_with_ground(
            self,
            data: mjx.Data,
            geom_ids: jax.Array
        ) -> jax.Array:
        """Return a jnp.bool_ indicating whether any geoms in geom_ids are in contact with the ground."""
        # Only consider contacts with distance <= tol
        tol = 0.001  # 1mm tolerance
        mask = data._impl.contact.dist <= tol
        indices = jnp.where(mask, size=mask.shape[0])[0]
        geom = data._impl.contact.geom[indices]

        geom_ids = jnp.array(geom_ids, dtype=geom.dtype)
        floor_gid = jnp.array(self._floor_gid, dtype=geom.dtype)

        # For each contact entry, check if either (geom[:,0] is the geom and geom[:,1] is floor)
        # or (geom[:,1] is the geom and geom[:,0] is floor). Then reduce with any(). This
        # naturally handles the empty-contact case (any over empty -> False).
        match1 = jnp.any(jnp.any(geom[:, 0, None] == geom_ids[None, :], axis=1) & (geom[:, 1] == floor_gid))
        match2 = jnp.any(jnp.any(geom[:, 1, None] == geom_ids[None, :], axis=1) & (geom[:, 0] == floor_gid))
        return jnp.logical_or(match1, match2)

    def _vector_com_to_support(self, data: mjx.Data) -> jax.Array:
        """
        Compute vector from COM projection to closest point on support geometry.

        Returns a 2D vector (COM_xy -> closest point on support). 
        If no feet contact the ground, returns zero vector.
        """
        # Foot centers in world frame
        left_center = jnp.array(data.xpos[self._left_foot_id])
        right_center = jnp.array(data.xpos[self._right_foot_id])

        # Check contact booleans
        left_foot_contact = self._is_in_contact_with_ground(data, self._left_foot_gid)
        right_foot_contact = self._is_in_contact_with_ground(data, self._right_foot_gid)

        com_xy = jnp.array(data.subtree_com[0][:2])

        # Case 1: both feet on the ground (support is a line segment)
        a = left_center[:2]
        b = right_center[:2]
        ab = b - a
        denom = jnp.dot(ab, ab) + 1e-12
        t = jnp.clip(jnp.dot(com_xy - a, ab) / denom, 0.0, 1.0)
        proj = a + t * ab
        vec_both = proj - com_xy

        # Case 2: only left foot on the ground (support is a point)
        vec_left = left_center[:2] - com_xy

        # Case 3: only right foot on the ground (support is a point)
        vec_right = right_center[:2] - com_xy

        # Case 4: no feet on the ground
        vec_none = jnp.zeros(2)

        # Choose result without Python control flow so the function is jittable.
        # Order: if both -> vec_both, else if left -> vec_left, else if right -> vec_right, else zero.
        both = jnp.logical_and(left_foot_contact, right_foot_contact)
        left_only = jnp.logical_and(left_foot_contact, jnp.logical_not(right_foot_contact))
        right_only = jnp.logical_and(right_foot_contact, jnp.logical_not(left_foot_contact))

        res = jnp.where(
            both, vec_both, jnp.where(
                left_only, vec_left, jnp.where(
                    right_only, vec_right, vec_none
                )
            )
        )
        return res
    