import numpy as np
from cassie_arl.rl_env.utility import euler2quat

'''
To randomize the dynamics parameters which will be used in the cassie_env
'''

class DomainRandomizer:
    def __init__(self, sim, sim_freq):
        self.sim_freq = sim_freq
        self.state_buffer_size = 0
        self.action_buffer_size = 0
        self.defualt_floor_friction = [1.0, 5e-3, 1e-4]
        self.rand_floor_friction = [1.0, 5e-3, 1e-4]
        self.nv = sim.nv
        self.nbody = sim.nbody
        self.default_damping = sim.get_dof_damping()  # 32(6+26)
        self.default_stiffness = sim.get_dof_stiffness()
        self.default_mass = sim.get_body_mass()  # 26
        self.default_ipos = (
            sim.get_body_ipos()
        )  # 78(26*3) xyz of COM of 26 parts of Cassie
        self.default_inertia = sim.get_body_inertia()
        self.__init_dynrand_range()
        self.reset()

    def reset(self):
        self.randomize_dynamics()

    def __init_dynrand_range(self):
        self.rand_range = 0.3
        self.fric_bound = np.array([0.3, 3.0])
        self.delay_bound = np.array([1.0 / 2200.0, 1.0 / 35.0])
        self.rot_noise_bound = np.array([-0.002, 0.002])
        self.rot_vel_noise_bound = np.array([-0.01, 0.01])
        self.lin_acc_noise_bound = np.array([-0.1, 0.1])
        self.lin_vel_noise_bound = np.array([-0.04, 0.04])
        self.pd_uncertainty_bound = np.array([1.0 - 0.3, 1.0 + 0.3])
        self.stiff_bound = np.array([1.0 - 0.2, 1.0 + 0.2])

    def randomize_dynamics(self):
        self.__randomlize_damping()
        self.__randomlize_stiffness()
        self.__randomlize_body_mass()
        self.__randomlize_mass_offset()
        self.__randomlize_delay()
        self.__randomlize_noise_magnitute()
        self.__randomlize_floor_friction()
        self.__randomlize_pd_uncertainty()
        self.__randomlize_body_inertia()
        self.__randomlize_floor_slope()

    def __randomlize_damping(self):
        rand_damp_ratio = np.random.uniform(1.0 - self.rand_range, 4.0, (self.nv,))
        self.rand_damping = np.clip(rand_damp_ratio * self.default_damping, 0, None)
        # self.sim.set_dof_damping(self.damping)

    def __randomlize_stiffness(self):
        rand_stiff_ratio = np.random.uniform(
            self.stiff_bound[0], self.stiff_bound[1], (self.nv,)
        )
        self.rand_stiffness = np.clip(
            rand_stiff_ratio * self.default_stiffness, 0, None
        )

    def __randomlize_body_mass(self):
        self.rand_mass = self.default_mass.copy()
        rand_mass_ratio = np.random.uniform(0.5, 1.5, (self.nbody,))
        self.rand_mass = rand_mass_ratio * self.default_mass

    def __randomlize_body_inertia(self):
        self.rand_inertia = self.default_inertia.copy()
        rand_inertia_ratio = np.random.uniform(0.7, 1.3, (self.nbody * 3,))
        self.rand_inertia = rand_inertia_ratio * self.default_inertia

    def __randomlize_mass_offset(self):
        rand_ipos_ratio = np.random.uniform(
            1.0 - self.rand_range, 1.0 + self.rand_range, (3 * self.nbody,)
        )
        self.rand_ipos = rand_ipos_ratio * self.default_ipos
        base_offset = np.random.uniform(-0.1, 0.1, (3,))
        general_link_ipos_offset = np.random.uniform(
            -0.01, 0.01, (3 * (self.nbody - 2),)
        )
        self.rand_ipos[[3, 4, 5]] += base_offset
        self.rand_ipos[6:] += general_link_ipos_offset

    def __randomlize_floor_friction(self):
        self.rand_floor_friction = np.copy(self.defualt_floor_friction)
        self.rand_floor_friction[0] = np.random.uniform(
            self.fric_bound[0], self.fric_bound[1]
        )
        self.rand_floor_friction[1] = np.random.uniform(
            0.0, self.defualt_floor_friction[1] * 3.0
        )
        self.rand_floor_friction[2] = np.random.uniform(
            0.0, self.defualt_floor_friction[2] * 3.0
        )

    def __randomlize_delay(self):
        t_delay = np.random.uniform(
            self.delay_bound[0], self.delay_bound[1]
        )  # delay from 2000 hz to 40 hz, udp will be running at 2000Hz realtime
        self.state_buffer_size = int(round(t_delay * self.sim_freq))
        t_action_delay = np.random.uniform(self.delay_bound[0], self.delay_bound[1])
        self.action_buffer_size = int(round(t_action_delay * self.sim_freq))

    def __randomlize_noise_magnitute(self):
        self.noise_rot = np.random.uniform(
            self.rot_noise_bound[0], self.rot_noise_bound[1]
        )
        self.noise_rot_vel = np.random.uniform(
            self.rot_vel_noise_bound[0], self.rot_vel_noise_bound[1]
        )
        self.noise_lin_acc = np.random.uniform(
            self.lin_acc_noise_bound[0], self.lin_acc_noise_bound[1]
        )
        self.noise_lin_vel = np.random.uniform(
            self.lin_vel_noise_bound[0], self.lin_vel_noise_bound[1]
        )

    def __randomlize_pd_uncertainty(self):
        self.pd_uncertainty = np.random.uniform(
            self.pd_uncertainty_bound[0], self.pd_uncertainty_bound[1], (20,)
        )  # for pd gains of 10 motors

    def __randomlize_floor_slope(self):
        flat_ground_flag = np.random.choice([True, False], p=[0.1, 0.9])
        if flat_ground_flag:
            self.floor_slope = np.zeros((3,))
        else:
            self.floor_slope = np.random.uniform(-0.03, 0.03, (3,))

    def get_rand_damping(self):
        return np.copy(self.rand_damping)

    def get_rand_stiffness(self):
        return np.copy(self.rand_stiffness)

    def get_rand_ipos(self):
        return np.copy(self.rand_ipos)

    def get_rand_mass(self):
        return np.copy(self.rand_mass)

    def get_rand_inertia(self):
        return np.copy(self.rand_inertia)

    def get_rand_floor_friction(self):
        return np.copy(self.rand_floor_friction)

    def get_rand_nosie(self):
        return (
            np.copy(self.noise_rot),
            np.copy(self.noise_rot_vel),
            np.copy(self.noise_lin_acc),
            np.copy(self.noise_lin_vel),
        )

    def get_state_buffer_size(self):
        return self.state_buffer_size

    def get_action_buffer_size(self):
        return self.action_buffer_size

    def get_pd_uncertainty(self):
        return np.copy(self.pd_uncertainty)

    def get_floor_slope(self):
        return euler2quat(self.floor_slope.reshape)
    