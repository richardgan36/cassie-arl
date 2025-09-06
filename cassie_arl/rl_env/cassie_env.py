from collections import deque

from gymnasium import spaces

# from rlenv.cassiemujoco import *
# from rlenv.cassiemujoco_ctypes import *
# from rlenv.reference_generator import ReferenceGenerator  # Probably won't need this for quiet standing
from cassie_arl.rl_env.action_filter import ActionFilterButter
from cassie_arl.rl_env.domain_randomizer import DomainRandomizer
# from rlenv.perturbation_generator import PerturbationGenerator  # Probably won't need this as we have an adversary, but may be useful for initial training / warmup
from cassie_arl.rl_env.cassie_fk import CassieFK
from cassie_arl.config.defaults import *
# from utility.utility import *


class CassieEnv:
    def __init__(
        self,
        config,
    ):
        self.max_timesteps = config["max_timesteps"]
        self.action_bounds = np.copy(ACTUATED_JOINT_RANGE)
        self.safe_action_bounds = np.copy(self.action_bounds)
        # for safe joint limits on hardware
        self.safe_action_bounds[:, [0, 5]] *= 0.5
        self.safe_action_bounds[:, [1, 6]] *= 0.8
        self.base_idx = [0, 1, 2, 3, 4, 5, 6]
        self.motor_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.motor_vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        self.pGain = np.copy(DEFAULT_PGAIN)
        self.dGain = np.copy(DEFAULT_DGAIN)
        self.offset_footJoint2midFoot = np.sqrt(0.01762**2 + 0.05219**2)
        self.sim = CassieSim(ROBOT_MODEL_FILE)
        if config["is_visual"]:
            self.is_visual = True
            self.vis = CassieVis(self.sim, ROBOT_MODEL_FILE)
            if config["cam_track_robot"]:
                self.vis.set_cam("cassie-pelvis", 2.5, 90, 0)
        else:
            self.is_visual = False
        self.step_zerotorque = config["step_zerotorque"]
        self.cassie_fk = CassieFK()
        # init variables
        self.qpos = np.copy(self.sim.qpos())
        self.qvel = np.copy(self.sim.qvel())
        self.qacc = np.copy(self.sim.qacc())
        self.obs_cassie_state = state_out_t()
        self.cassie_out = cassie_out_t()
        self.u = self._init_u()
        self.num_motor = 10
        self.sim_freq = 2000
        self.appx_env_freq = 30  # 30 hz but the real control frequency is not exactly 30Hz because we round up the num_sims_per_env_step
        self.num_sims_per_env_step = self.sim_freq // self.appx_env_freq
        self.secs_per_env_step = self.num_sims_per_env_step / self.sim_freq
        self.real_env_freq = int(1 / self.secs_per_env_step)
        self.history_len_vf = 4
        self.history_len_pol = 2 * self.real_env_freq
        # observation
        self.observation_space_pol = None
        self.observation_space_vf = None 
        self.observation_space_pol_cnn = None
        self.extrinsics_dim = None
        # the low & high does not actually limit the actions output from MLP network, manually clip instead
        self.action_space = spaces.Box(low=-100, high=100, shape=(self.num_motor,))
        self.previous_obs = deque(maxlen=self.history_len_vf)
        self.previous_acs = deque(maxlen=self.history_len_vf)
        self.long_history = deque(maxlen=self.history_len_pol)
        # reference motion
        self.reference_generator = ReferenceGenerator(
            env_max_timesteps=self.max_timesteps,
            secs_per_env_step=self.secs_per_env_step,
            config=config,
        )
        # action filter
        self.action_filter_order = 2
        self.__init_action_filter()
        # dynamics randomization
        self.__set_env_type(config)
        # set reward
        self.__init_reward_func()
        self.__init_env_randomlizer()
        self.__init_perturbation_generator()
        # init step
        self.__init_step_func()
        # reset and init others
        self.reset()
        # self.sim.set_geom_name_pos('table', [0.6, 0.85, -0.265]) # 0.5 is the max

    ##########################################
    #            Init and Reset              #
    ##########################################
    def __set_env_type(self, config):
        # NOTE: set true to use minimal rand range -> only rand floor friction
        self.minimal_rand = config["minimal_rand"]
        # NOTE: set true to add noise on observation
        self.noisy = config["is_noisy"]  
        # NOTE: set true to add external perturbation
        self.perturbation = config["add_perturbation"]

    def __init_reward_func(self):
        self.reward_names = [
            "r_mpos",
            "r_mvel",
            "r_ptrans",
            "r_ptrans_velx",
            "r_ptrans_vely",
            "r_prot",
            "r_prot_vel",
            "r_ptrans_z",
            "r_torque",
            "r_foot_force",
            "r_acc",
            "r_footpos",
            "r_delta_acs",
        ]
        w_mpos = 15.0
        w_mvel_nonstand = 0.0
        w_mvel_stand = 15.0
        w_ptrans_pos = 6.0
        w_ptrans_velx = 7.5
        w_ptrans_vely = 7.5
        w_prot_pos = 12.5
        w_prot_vel = 3.0
        w_ptrans_z = 5.0
        w_torque = 3.0
        w_foot_force = 10.0
        w_acc = 3.0
        w_footpos = 3.0
        w_delta_acs_nonstand = 3.0
        w_delta_acs_stand = 10.0

        w_array_nonstand = np.array(
            [
                w_mpos,
                w_mvel_nonstand,
                w_ptrans_pos,
                w_ptrans_velx,
                w_ptrans_vely,
                w_prot_pos,
                w_prot_vel,
                w_ptrans_z,
                w_torque,
                w_foot_force,
                w_acc,
                w_footpos,
                w_delta_acs_nonstand,
            ]
        )
        w_array_stand = np.array(
            [
                w_mpos,
                w_mvel_stand,
                w_ptrans_pos,
                w_ptrans_velx,
                w_ptrans_vely,
                w_prot_pos,
                w_prot_vel,
                w_ptrans_z,
                w_torque,
                w_foot_force,
                w_acc,
                w_footpos,
                w_delta_acs_stand,
            ]
        )

        self.reward_weights = w_array_nonstand / np.sum(w_array_nonstand)
        self.reward_weights_stand = w_array_stand / np.sum(w_array_stand)
        self.reward_scales = dict()
        # to balance unit: meter/radians/N/Nm/etc, for reward: exp(-scale*norm)
        self.reward_scales["mpos"] = 5.0
        self.reward_scales["mvel"] = 1e-2
        self.reward_scales["mvel_stand"] = 50.0
        self.reward_scales["prot"] = 10.0
        self.reward_scales["prot_vel"] = 1.0
        self.reward_scales["meter"] = 3.0
        self.reward_scales["meter_z"] = 10.0
        self.reward_scales["meter_velx"] = 2.5
        self.reward_scales["meter_vely"] = 5.0
        self.reward_scales["torque"] = 5e-7
        self.reward_scales["foot_force"] = 2.5e-5
        self.reward_scales["double_foot_force"] = 1.5e-5
        self.reward_scales["side_foot_force"] = 1.5e-3
        self.reward_scales["ppos_acc"] = 2e-5
        self.reward_scales["prot_acc"] = 1e-5
        self.reward_scales["macc"] = 1e-9
        self.reward_scales["foot_pos"] = 100
        self.reward_scales["delta_acs"] = 2.0

        # add a negative sign
        for key in self.reward_scales:
            self.reward_scales[key] *= -1.0

    def __init_step_func(self):
        if self.step_zerotorque:
            self.__sim_step = self.__step_sim_zerotorque
        else:
            self.__sim_step = self.__step_sim_nominal
        self.__get_pTargets = self.acs_norm2actual

    def _init_u(self):
        # p_paras = [400, 200, 200, 500, 20]
        # d_paras = [4, 4, 10, 20, 4]
        # feed forward torque
        torque_fwd = [0, 0, 0, 0, 0]
        self.pd_uncertainty = np.ones((20,))
        # init controller class u
        u = pd_in_t()
        # assign PD controller parameters
        for i in range(5):
            # left leg
            u.leftLeg.motorPd.torque[i] = 0  # torque_fwd[i]  # Feedforward torque
            u.leftLeg.motorPd.pTarget[i] = 0  # p_target[i]
            u.leftLeg.motorPd.pGain[i] = self.pGain[i]
            u.leftLeg.motorPd.dTarget[i] = 0
            u.leftLeg.motorPd.dGain[i] = self.dGain[i]
        u.rightLeg.motorPd = u.leftLeg.motorPd
        return u

    def _update_pd(self):
        self.u.leftLeg.motorPd.pGain[:5] = self.pGain[:5] * self.pd_uncertainty[:5]
        self.u.leftLeg.motorPd.dGain[:5] = self.dGain[:5] * self.pd_uncertainty[10:15]
        self.u.rightLeg.motorPd.pGain[:5] = self.pGain[5:10] * self.pd_uncertainty[5:10]
        self.u.rightLeg.motorPd.dGain[:5] = self.dGain[5:10] * self.pd_uncertainty[15:20]

    def __init_robot_pos(self):
        self.reference_generator.reset()
        (
            ref_base_pos,
            ref_base_rot,
            ref_mpos,
        ) = self.reference_generator.get_init_pose()
        ref_q = euler2quat(ref_base_rot.reshape((3,)))
        ref_base = np.hstack([ref_base_pos.reshape((1, 3)), ref_q.reshape((1, 4))])
        self.set_motor_base_pos(ref_mpos, ref_base)
        self._init_obs_cassie_state(ref_mpos.ravel(), ref_base.ravel())

    def _init_obs_cassie_state(self, ref_mpos, ref_base):
        cassie_out = self.__set_cassie_out(ref_mpos, ref_base)
        self.obs_cassie_state = self.sim.estimate_state(cassie_out)
        self.__add_obs_noise()
        # print("init obs_cassie_state:", np.array(self.obs_cassie_state.pelvis.translationalAcceleration).ravel())

    def __init_action_filter(self):
        self.action_filter = ActionFilterButter(
            lowcut=None,
            highcut=[4],
            sampling_rate=self.real_env_freq,
            order=self.action_filter_order,
            num_joints=self.num_motor,
        )

    def __init_env_randomlizer(self):
        self.env_randomlizer = DomainRandomizer(self.sim, self.sim_freq)
        self.__set_dynamics_properties()

    def __init_perturbation_generator(self):
        self.pertubration_generator = PerturbationGenerator(self.real_env_freq, 0.2)

    def __reset_action_filter(self):
        self.action_filter.reset()
        noise_rot_magnitute, _, _, _ = self.env_randomlizer.get_rand_nosie()
        default_action = (
            self.qpos[self.motor_idx]
            + np.random.normal(size=self.num_motor) * noise_rot_magnitute
        )
        self.action_filter.init_history(default_action)

    def reset(self):
        self.__reset_env()
        self.action_filter.reset()
        self.pertubration_generator.reset()
        self.env_randomlizer.randomize_dynamics()
        self.__set_dynamics_properties()
        self.__update_data(step=False)
        self.__reset_consts()
        self.__reset_action_filter()
        obs_vf, obs_pol = self.__get_observation(step=False)
        if self.observation_space_vf is None:
            self.__set_obs_space(obs_vf, obs_pol)
        return obs_vf, obs_pol

    def __reset_env(self):
        self.sim.full_reset()
        self.sim.set_time(0)
        self.timestep = 0
        self.time_in_sec = 0.0
        self.previous_obs = deque(maxlen=self.history_len_vf)
        self.previous_acs = deque(maxlen=self.history_len_vf)
        self.long_history = deque(maxlen=self.history_len_pol)

        self.__init_robot_pos()

        self.reward = None
        self.done = None
        self.info = {}
        self.fall_flag = False

    def __reset_consts(self):
        self.init_xy = self.qpos[[0, 1]]
        self.height = self.qpos[2]
        self.foot_pos = np.empty(6)
        self.sim.foot_pos(self.foot_pos)
        self.init_foot_pos = self.foot_pos
        self.applied_force = np.zeros((6,))
        self.last_acs = self.qpos[self.motor_idx]

    def __set_dynamics_properties(self):
        if self.minimal_rand:
            self.sim.set_geom_friction(
                self.env_randomlizer.get_rand_floor_friction(), "floor"
            )
        else:
            self.sim.set_dof_damping(self.env_randomlizer.get_rand_damping())
            self.sim.set_dof_stiffness(self.env_randomlizer.get_rand_stiffness())
            self.sim.set_body_mass(self.env_randomlizer.get_rand_mass())
            self.sim.set_body_ipos(self.env_randomlizer.get_rand_ipos())
            self.sim.set_body_inertia(self.env_randomlizer.get_rand_inertia())
            self.sim.set_geom_friction(
                self.env_randomlizer.get_rand_floor_friction(), "floor"
            )
            self.pd_uncertainty = self.env_randomlizer.get_pd_uncertainty()
            self._update_pd()
            # self.sim.set_geom_quat(self.env_randomlizer.get_floor_slope(), 'floor')

    def __set_obs_space(self, obs_vf, obs_pol):
        self.observation_space_vf = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_vf.shape
        )
        self.observation_space_pol = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_pol[0].shape
        )
        self.observation_space_pol_cnn = obs_pol[1].shape
        print("obs space has init!")

    ##########################################
    #                Step                    #
    ##########################################
    def __step_sim_nominal(self, actual_pTs_filtered):
        # 1 control_step = 0.0005s / 2kHz
        self.u.leftLeg.motorPd.pTarget[:5] = actual_pTs_filtered[:5]
        self.u.rightLeg.motorPd.pTarget[:5] = actual_pTs_filtered[5:]
        for _ in range(self.num_sims_per_env_step):
            # give pTargets to motors
            self.obs_cassie_state = self.sim.estimate_state(
                self.sim.step_pd_without_estimation(self.u)
            )

    def __step_sim_zerotorque(self, actual_pTs_filtered):
        # 1 control_step = 0.0005s / 2kHz
        self.u.leftLeg.motorPd.pTarget[:5] = 0.0 * actual_pTs_filtered[:5]
        self.u.rightLeg.motorPd.pTarget[:5] = 0.0 * actual_pTs_filtered[5:]
        for _ in range(self.num_sims_per_env_step):
            # give pTargets to motors
            self.obs_cassie_state = self.sim.estimate_state(
                self.sim.step_pd_without_estimation(self.u)
            )

    def step(self, acs, restore=False):
        """
        :param act: a dict {control index: pTarget, ...}
        :return:
        """
        assert acs.shape[0] == self.num_motor and -1.0 <= acs.all() <= 1.0
        actual_pTs = self.__get_pTargets(acs)
        actual_pTs_filtered = self.action_filter.filter(actual_pTs)

        if self.perturbation:
            self.__apply_perturbation()
        # simulated env
        self.__sim_step(actual_pTs_filtered)
        if self.noisy:
            self.__add_obs_noise()

        self.__update_data(step=True)
        obs_vf, obs_pol = self.__get_observation(acs=actual_pTs, step=True)
        reward, reward_dict = self.__get_reward(acs=actual_pTs)
        done = self.__is_done() if not restore else False
        self.info["reward_dict"] = reward_dict
        self.last_acs = actual_pTs
        return obs_vf, obs_pol, reward, done, self.info

    def __update_data(self, step=True):
        self.qpos = np.copy(self.sim.qpos())
        self.qvel = np.copy(self.sim.qvel())
        self.qacc = np.copy(self.sim.qacc())
        self.height = self.qpos[2]
        self.foot_pos = np.empty(6)
        self.sim.foot_pos(self.foot_pos)
        self.curr_rpy_gt = quat2euler(self.qpos[3:7])
        self.curr_rpy_obs = quat2euler(self.obs_cassie_state.pelvis.orientation)
        if step:
            self.timestep += 1
            self.time_in_sec = (
                self.timestep * self.num_sims_per_env_step
            ) / self.sim_freq
            self.reference_generator.update_ref_env(
                self.time_in_sec, self.qpos[:2], self.curr_rpy_gt[-1]
            )
        self.ref_dict = self.reference_generator.get_ref_motion()

    def __add_obs_noise(self):
        noise_vec = np.random.normal(size=(32,))  # 20+3+3+3+3
        (
            noise_rot,
            noise_rot_vel,
            noise_linear_acc,
            noise_linear_vel,
        ) = self.env_randomlizer.get_rand_nosie()
        obs_euler = quat2euler(self.obs_cassie_state.pelvis.orientation)
        self.obs_cassie_state.pelvis.translationalVelocity[:3] += (
            noise_vec[:3] * noise_linear_vel
        )
        self.obs_cassie_state.pelvis.translationalAcceleration[:3] += (
            noise_vec[3:6] * noise_linear_acc
        )
        self.obs_cassie_state.pelvis.rotationalVelocity[:3] += (
            noise_vec[6:9] * noise_rot_vel
        )
        obs_euler[:3] += noise_vec[9:12] * noise_rot
        obs_quat = euler2quat(obs_euler)
        self.obs_cassie_state.pelvis.orientation[:4] = obs_quat[:4]
        self.obs_cassie_state.motor.position[:10] += noise_vec[12:22] * noise_rot
        self.obs_cassie_state.motor.velocity[:10] += noise_vec[22:32] * noise_rot_vel

    def __set_cassie_out(self, mpos, base_pos):
        cassie_out = cassie_out_t()
        cassie_out.pelvis.vectorNav.orientation[:4] = base_pos[3:7]
        # motor, leftleg
        cassie_out.leftLeg.hipRollDrive.position = mpos[0]
        cassie_out.leftLeg.hipYawDrive.position = mpos[1]
        cassie_out.leftLeg.hipPitchDrive.position = mpos[2]
        cassie_out.leftLeg.kneeDrive.position = mpos[3]
        cassie_out.leftLeg.footDrive.position = mpos[4]
        # motor, rightleg
        cassie_out.rightLeg.hipRollDrive.position = mpos[5]
        cassie_out.rightLeg.hipYawDrive.position = mpos[6]
        cassie_out.rightLeg.hipPitchDrive.position = mpos[7]
        cassie_out.rightLeg.kneeDrive.position = mpos[8]
        cassie_out.rightLeg.footDrive.position = mpos[9]
        return cassie_out

    def __apply_perturbation(self):
        (
            force_to_apply,
            apply_force_flag,
        ) = self.pertubration_generator.apply_perturbation()
        if apply_force_flag:
            self.applied_force = force_to_apply
            self.sim.apply_force(self.applied_force)
            if self.is_visual:
                print("Applied Perturbation: ", self.applied_force)
        else:
            self.applied_force *= 0.0
            self.sim.clear_forces()

    ##########################################
    #              Observation               #
    ##########################################
    def __get_observation(self, acs=np.zeros(10), step=False):
        ref_dict_1 = self.reference_generator.get_ref_motion(look_forward=1)
        ref_dict_4 = self.reference_generator.get_ref_motion(look_forward=4)
        ref_dict_7 = self.reference_generator.get_ref_motion(look_forward=7)

        ob1 = ref_dict_1["motor_pos"]
        ob4 = ref_dict_4["motor_pos"]
        ob7 = ref_dict_7["motor_pos"]

        ob_curr = np.concatenate(
            [
                np.array(self.obs_cassie_state.pelvis.translationalVelocity).ravel(),
                np.array(self.obs_cassie_state.pelvis.orientation).ravel(),
                np.array(self.obs_cassie_state.motor.position).ravel(),
                np.array(self.obs_cassie_state.motor.velocity).ravel(),
            ]
        )

        # ground truth observation
        curr_xy_gt = self.qpos[[0, 1]] - self.init_xy
        ob_curr_gt = np.concatenate(
            [
                self.qvel[[0, 1, 2]],
                self.qacc[[0, 1, 2]],
                self.qpos[3:7],
                self.qpos[self.motor_idx],
                self.qvel[self.motor_vel_idx],
            ]
        )

        # command
        ref_yaw = self.ref_dict["base_rot_global"][-1]
        ob_command = np.concatenate(
            [
                [self.ref_dict["base_pos_global"][-1]],  # walking height
                self.ref_dict["base_vel_local"][[0, 1]],  # local velocity command
                [math.cos(ref_yaw), math.sin(ref_yaw)],  # desired turning yaw angle
            ]
        )  #  height, vx, vy, yaw
        if self.timestep == 0:
            [self.previous_obs.append(ob_curr) for i in range(self.history_len_vf)]
            [
                self.previous_acs.append(np.zeros(self.num_motor))
                for _ in range(self.history_len_vf)
            ]
            [
                self.long_history.append(
                    np.concatenate([ob_curr, np.zeros(self.num_motor)])
                )
                for _ in range(self.history_len_pol)
            ]
        ob_prev = np.concatenate(
            [np.array(self.previous_obs).ravel(), np.array(self.previous_acs).ravel()]
        )
        obs_pol_hist = np.flip(np.asarray(self.long_history).T, 1)

        obs_pol_base = np.concatenate([ob_prev, ob_curr, ob1, ob4, ob7, ob_command])
        obs_pol = (obs_pol_base, obs_pol_hist)

        obs_vf = np.concatenate(
            [
                ob_prev,
                ob_curr_gt,
                ob1,
                ob4,
                ob7,
                ob_command,
                self.ref_dict["base_pos_global"][[0, 1]] - curr_xy_gt,
                np.array([self.qpos[2]]),
                self.sim.get_foot_forces(),
                self.env_randomlizer.get_rand_floor_friction(),
            ]
        )

        if step:
            self.previous_obs.append(ob_curr)
            self.previous_acs.append(acs)
            self.long_history.append(np.concatenate([ob_curr, acs]))
        return obs_vf, obs_pol

    ##########################################
    #                Reward                  #
    ##########################################
    def __get_reward(self, acs):
        # NOTE: reward is using qpos/qvel that don't have noise and delay
        mpos_err = np.sum(
            np.square(self.ref_dict["motor_pos"] - self.qpos[self.motor_idx])
        )
        r_mpos = np.exp(self.reward_scales["mpos"] * mpos_err)

        mvel_err = np.sum(
            np.square(self.ref_dict["motor_vel"] - self.qvel[self.motor_vel_idx])
        )
        if not self.reference_generator.in_stand_mode:
            r_mvel = np.exp(self.reward_scales["mvel"] * mvel_err)
        else:
            r_mvel = np.exp(self.reward_scales["mvel_stand"] * mvel_err)

        curr_xy = self.qpos[[0, 1]] - self.init_xy
        base_pos_err = np.sum(
            np.square(self.ref_dict["base_pos_global"][[0, 1]] - curr_xy)
        )  # pos global
        r_ptrans = np.exp(self.reward_scales["meter"] * base_pos_err)

        velocity_local = global2local(self.qvel[:3], self.curr_rpy_gt)

        r_ptrans_velx = np.exp(
            self.reward_scales["meter_velx"]
            * abs(self.ref_dict["base_vel_local"][0] - velocity_local[0])
        )
        # print(
        #     "ptrans_velx reward: {:.6f}, desired: {:.3f}, actual: {:.3f}".format(
        #         r_ptrans_velx, self.ref_dict["base_vel_local"][0], velocity_local[0]
        #     ),
        #     end="\r",
        # )

        r_ptrans_vely = np.exp(
            self.reward_scales["meter_vely"]
            * abs(self.ref_dict["base_vel_local"][1] - velocity_local[1])
        )
        # print(
        #     "ptrans_vely reward: {:.6f}, desired: {:.3f}, actual: {:.3f}".format(
        #         r_ptrans_vely, self.ref_dict["base_vel_local"][1], velocity_local[1]
        #     ),
        #     end="\r",
        # )

        base_angle_err = np.sum(
            1.0 - np.cos(self.ref_dict["base_rot_global"] - self.curr_rpy_gt)
        )
        r_prot = np.exp(self.reward_scales["prot"] * base_angle_err)
        # print(
        #     "prot reward:{:.6f}, desired: {:.4f}, actual: {:.4f}".format(
        #         r_prot,
        #         np.rad2deg(self.ref_dict["base_rot_global"][-1]),
        #         np.rad2deg(self.curr_rpy_gt[-1]),
        #     ),
        #     end="\r",
        # )

        base_anglevel_err = np.sum(
            np.square(
                self.qvel[[3, 4, 5]]
                - np.array([0.0, 0.0, self.ref_dict["base_vel_local"][-1]])
            )
        )  # stablize roll and pitch and track yaw vel
        r_prot_vel = np.exp(self.reward_scales["prot_vel"] * base_anglevel_err)

        ptrans_z_err = np.square(self.qpos[2] - self.ref_dict["base_pos_global"][2])
        r_ptrans_z = np.exp(self.reward_scales["meter_z"] * ptrans_z_err)

        torques_err = np.sum(np.square(self._calc_torque(acs)))
        r_torques = np.exp(self.reward_scales["torque"] * torques_err)

        impact_forces = self.sim.get_foot_impact_forces()
        if self.reference_generator.in_stand_mode:
            foot_forces = self.sim.get_foot_forces()
            foot_forces_side = foot_forces[[0, 1, 3, 4]]
            if any(abs(impact_forces) <= 1.0):
                r_foot_force = 0.0
            else:
                foot_force_err = np.sum(np.square(foot_forces_side))
                r_foot_force = np.exp(
                    self.reward_scales["side_foot_force"] * foot_force_err
                )
        else:
            if abs(impact_forces[0]) <= 0.1 and abs(impact_forces[1]) <= 0.1:
                r_foot_force = 0.0
            elif abs(impact_forces[0]) >= 0.1 and abs(impact_forces[1]) >= 0.1:
                foot_force_err = np.square(np.sum(impact_forces) / 2.0)
                r_foot_force = np.exp(
                    self.reward_scales["double_foot_force"] * foot_force_err
                )
            else:
                foot_force_err = np.sum(np.square(impact_forces))
                r_foot_force = np.exp(self.reward_scales["foot_force"] * foot_force_err)

        ref_footpos = self.get_foot_pos_absolute(
            self.ref_dict["base_pos_global"],
            self.ref_dict["base_rot_global"],
            self.ref_dict["motor_pos"],
        )
        foot_pos_err = np.sum(
            np.square(
                ref_footpos[[2, 5]]
                - self.offset_footJoint2midFoot
                - self.foot_pos[[2, 5]]
            )
        )
        r_footpos = np.exp(self.reward_scales["foot_pos"] * foot_pos_err)

        base_xyz_acc_err = np.sum(np.square(self.qacc[0:3]))
        base_rxyz_acc_err = np.sum(np.square(self.qacc[3:6]))
        base_motor_acc_err = np.sum(np.square(self.qacc[self.motor_vel_idx]))
        r_base_xyz_acc = np.exp(self.reward_scales["ppos_acc"] * base_xyz_acc_err)
        r_base_rxyz_acc = np.exp(self.reward_scales["prot_acc"] * base_rxyz_acc_err)
        r_motor_acc = np.exp(self.reward_scales["macc"] * base_motor_acc_err)
        r_acc = (r_base_xyz_acc + r_base_rxyz_acc + r_motor_acc) / 3.0

        r_delta_acs = np.exp(
            self.reward_scales["delta_acs"] * np.sum(np.square(self.last_acs - acs))
        )

        # NOTE: should be in the same order with self.reward_weights
        rewards = np.array(
            [
                r_mpos,
                r_mvel,
                r_ptrans,
                r_ptrans_velx,
                r_ptrans_vely,
                r_prot,
                r_prot_vel,
                r_ptrans_z,
                r_torques,
                r_foot_force,
                r_acc,
                r_footpos,
                r_delta_acs,
            ]
        )

        if not self.reference_generator.in_stand_mode:
            total_reward = np.sum(self.reward_weights * rewards)
            reward_dict = dict(zip(self.reward_names, self.reward_weights * rewards))
        else:
            total_reward = np.sum(self.reward_weights_stand * rewards)
            reward_dict = dict(
                zip(self.reward_names, self.reward_weights_stand * rewards)
            )
        return total_reward, reward_dict

    ##########################################
    #           Early Termination            #
    ##########################################
    def __is_done(self):
        tarsus_pos = self.get_tarsus_pos(
            self.qpos[[0, 1, 2]], self.qpos[4:7], self.qpos[self.motor_idx]
        )
        if self.height < FALLING_THRESHOLD:
            self.fall_flag = True
            # print('below loweset height')
            return True
        elif any(tarsus_pos[[2, 5]] <= TARSUS_HITGROUND_THRESHOLD):
            self.fall_flag = True
            # print('tarsus on the ground')
            return True
        elif self.timestep >= self.max_timesteps:
            # print('max step reached:{}'.format(self.max_timesteps))
            return True
        else:
            return False

    ##########################################
    #                 Utils                  #
    ##########################################
    def acs_actual2norm(self, actual_acs):
        return (actual_acs - self.safe_action_bounds[0]) * 2 / (
            self.safe_action_bounds[1] - self.safe_action_bounds[0]
        ) - 1

    def acs_norm2actual(self, acs):
        return self.safe_action_bounds[0] + (acs + 1) / 2.0 * (
            self.safe_action_bounds[1] - self.safe_action_bounds[0]
        )

    def set_motor_base_pos(self, motor_pos, base_pos, iters=1500):
        """
        Kind of hackish.
        This takes a floating base position and some joint positions
        and abuses the MuJoCo solver to get the constrained forward
        kinematics.
        There might be a better way to do this, e.g. using mj_kinematics
        """
        for _ in range(iters):
            qpos = np.copy(self.sim.qpos())
            qvel = np.copy(self.sim.qvel())

            qpos[self.motor_idx] = motor_pos
            qpos[self.base_idx] = base_pos

            self.sim.set_qpos(qpos)
            self.sim.set_qvel(0 * qvel)

            self.sim.step_pd_without_estimation(pd_in_t())
            self.obs_cassie_state = self.sim.estimate_state(cassie_out_t())
            self.sim.set_time(self.time_in_sec)

    def close(self):
        if self.is_visual:
            self.vis.__del__()
        self.sim.__del__()

    def render(self):
        return self.vis.draw(self.sim)

    def _calc_torque(self, actual_pTs):
        torques = self.pGain * (
            actual_pTs - np.array(self.obs_cassie_state.motor.position)
        ) - self.dGain * np.array(self.obs_cassie_state.motor.velocity)
        return torques

    def get_foot_pos_relative(self, base_rot, motor_pos):
        return self.cassie_fk.get_foot_pos(motor_pos, [0, 0, 0], base_rot)

    def get_foot_pos_absolute(self, base_pos, base_rot, motor_pos):
        return self.cassie_fk.get_foot_pos(motor_pos, base_pos, base_rot)

    def get_tarsus_pos(self, base_pos, base_rot, motor_pos):
        return self.cassie_fk.get_tarsus_pos(motor_pos, base_pos, base_rot)
    