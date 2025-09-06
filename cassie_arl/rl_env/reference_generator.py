import numpy as np
from rlenv.gait_library import GaitLibrary
from rlenv.cmd_generator import CmdGenerateor
from configs.defaults import STANDING_POSE

'''
A generator for reference motion, which will be used in cassie_env.py. 
The env script only look at the ref_dict returned from this class. 
ref_dict contains all the info of the refernece motion, 
including reference motor position, motor velocity, base translational/rotational position/velocity
The ReferenceGenerator composes the data from cmd_generator.py & gait_libraray.py
and randomly set the robot to stand for a random timespan. 
'''


class ReferenceGenerator:
    def __init__(
        self,
        env_max_timesteps,
        secs_per_env_step,
        config,
    ):
        self.cmd_generator = CmdGenerateor(env_max_timesteps, secs_per_env_step, config)
        self.gait_library = GaitLibrary(secs_per_env_step)
        self.time_stand_transit_cooling = 3.0  # allow 3 sec to transit to standing
        self.norminal_standing = np.copy(STANDING_POSE)
        self.add_standing = config["add_standing"]
        self.reset()

    def reset(self):
        self.time_in_sec = 0.0
        self.cmd_generator.reset()
        self.gait_library.reset()
        if self.add_standing:
            stand_flag = np.random.choice([True, False], p=[0.9, 0.1])
            if stand_flag:
                self.time_standing_start = np.random.uniform(5.0, 30.0)
            else:
                self.time_standing_start = 10000.0
        else:
            self.time_standing_start = 10000.0
        self.standing_span = np.random.uniform(2.0, 30.0)
        self.start_standing = False
        self.end_standing = False
        self.last_ref_gaitparams = np.array([0.0, 0.0, 0.98])
        self.last_ref_rotparams = np.array([0.0, 0.0, 0.0])
        init_stand_flag = np.random.choice([True, False], p=[0.5, 0.5])
        if init_stand_flag:
            self.init_standing_flag = True  # stand at the first time
            self.last_standing_flag = False
        else:
            self.init_standing_flag = False  # jump at the first time
            self.last_standing_flag = True

    def update_ref_env(self, time_in_sec, base_xy_g, base_yaw):
        self.time_in_sec = time_in_sec
        if (
            not self.start_standing
            and self.time_in_sec >= self.time_standing_start
            and self.time_in_sec < self.time_standing_start + self.standing_span
        ):
            self.start_standing = True
        if (
            self.start_standing
            and self.time_in_sec >= self.time_standing_start + self.standing_span
        ):
            self.end_standing = True
            self.start_standing = False
            self.cmd_generator.clear_stand_mode()
        if self.start_standing:
            self.cmd_generator.start_stand_mode()
        self.cmd_generator.update_cmd_env(time_in_sec)
        ref_gaitparams = self.cmd_generator.curr_ref_gaitparams
        ref_rotparams = self.cmd_generator.curr_ref_rotcmds
        self.gait_library.update_gaitlib_env(
            gait_param=ref_gaitparams, time_in_sec=time_in_sec
        )
        if abs(self.last_ref_gaitparams[0] - ref_gaitparams[0]) >= 0.01:
            self.cmd_generator.set_ref_global_pos(xy=base_xy_g)
        if abs(self.last_ref_rotparams[-1] - ref_rotparams[-1]) >= 0.002:  # 0.1 deg
            self.cmd_generator.set_ref_global_yaw(yaw=base_yaw)
        self.last_ref_gaitparams = ref_gaitparams
        self.last_ref_rotparams = ref_rotparams

    def get_init_pose(self):
        # should reset ref env first then use this function
        init_gait_params = self.gait_library.get_random_init_gaitparams()
        ref_mpos = self.gait_library.get_ref_states(init_gait_params)
        ref_base_pos_from_cmd, _ = self.cmd_generator.get_ref_base_global()
        ref_base_pos = np.array(
            [
                ref_base_pos_from_cmd[0],
                ref_base_pos_from_cmd[1],
                init_gait_params[-1],
            ]
        )
        norminal_pose_flag = np.random.choice([True, False], p=[0.5, 0.5])
        ref_base_rot = np.array(
            [
                np.radians(np.random.uniform(-2.0, 2.0)),
                np.radians(np.random.uniform(-5.0, 5.0)),
                np.radians(np.random.uniform(-10.0, 10.0)),
            ]
        ).reshape((1, 3))
        if norminal_pose_flag:
            ref_base_pos, _, ref_mpos = self.norminal_pose
            stand_abduction_offset = np.radians(np.random.uniform(-1.0, 7.5, (2,)))
            ref_mpos[0] = ref_mpos[0] + stand_abduction_offset[0]
            ref_mpos[5] = ref_mpos[5] - stand_abduction_offset[1]
            stand_knee_offset = np.radians(np.random.uniform(-5.0, 5.0, (2,)))
            stand_thigh_offset = np.radians(np.random.uniform(-5.0, 5.0, (2,)))
            ref_mpos[3] += stand_knee_offset[0]
            ref_mpos[3 + 5] += stand_knee_offset[1]
            ref_mpos[2] += stand_thigh_offset[0]
            ref_mpos[2 + 5] += stand_thigh_offset[1]
            ref_base_pos[2] += np.random.uniform(-0.05, 0.05)
        return ref_base_pos, ref_base_rot, ref_mpos

    def get_ref_motion(self, look_forward=0):
        ref_dict = dict()
        ref_gait_params = self.cmd_generator.curr_ref_gaitparams
        ref_rot_params = self.cmd_generator.curr_ref_rotcmds
        (
            ref_base_pos_from_cmd,
            ref_base_rot_from_cmd,
        ) = self.cmd_generator.get_ref_base_global()
        ref_mpos = self.gait_library.get_ref_states(ref_gait_params, look_forward)
        ref_dict["base_pos_global"] = np.array(
            [*ref_base_pos_from_cmd, ref_gait_params[-1]]
        )
        ref_dict["base_rot_global"] = ref_base_rot_from_cmd
        ref_dict["base_vel_local"] = np.array(
            [ref_gait_params[0], ref_gait_params[1], ref_rot_params[-1]]
        )  # vx vy vyaw
        if self.start_standing:
            ref_dict["motor_pos"] = self.norminal_mpos
            ref_dict["motor_vel"] = np.zeros((10,))
        else:
            ref_dict["motor_pos"] = ref_mpos
            ref_dict["motor_vel"] = np.zeros((10,))  # ref_mvel
        return ref_dict

    def get_curr_params(self):
        ref_gait_params = self.cmd_generator.curr_ref_gaitparams
        ref_rot_params = self.cmd_generator.curr_ref_rotcmds
        return ref_gait_params, ref_rot_params

    @property
    def norminal_pose(self):
        ref_base_pos = self.norminal_base_pos
        ref_base_rot = self.norminal_base_rot
        ref_mpos = self.norminal_mpos
        return ref_base_pos, ref_base_rot, ref_mpos

    @property
    def norminal_base_pos(self):
        return np.copy(self.norminal_standing[[0, 1, 2]])

    @property
    def norminal_base_rot(self):
        return np.copy(self.norminal_standing[[3, 4, 5]])

    @property
    def norminal_mpos(self):
        return np.copy(self.norminal_standing[6:])

    @property
    def in_transit_to_stand(self):
        return (
            self.start_standing
            and self.time_in_sec
            <= self.time_standing_start + self.time_stand_transit_cooling
        )

    @property
    def in_stand_mode(self):
        return self.start_standing