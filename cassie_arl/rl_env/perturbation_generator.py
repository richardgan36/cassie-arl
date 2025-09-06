import numpy as np

'''
Produce random perturbation wrench lasting for a random timepspan with a random timespan for the next
'''

class PerturbationGenerator:
    def __init__(self, sim_freq, average_time):
        self.sim_freq = sim_freq
        self.apply_percent = average_time / self.sim_freq
        self.__init_range()
        self.reset()

    def __init_range(self):
        self.wrench_min = np.array([-20, -20, -20, -5, -5, -5]).reshape((6, 1))
        self.wrench_max = np.array([20, 20, 20, 5, 5, 5]).reshape((6, 1))

    def reset(self):
        self.apply_flag = False
        self.prev_apply_flag = False
        self.applied_force = np.zeros((6, 1))
        self.applied_time = 0
        self.to_apply_time = 0

    def apply_perturbation(self):
        # 0.5% to apply pertubation
        to_apply = np.random.choice(2, p=[self.apply_percent, 1 - self.apply_percent])
        if to_apply is 0:
            self.apply_flag = True
        if self.apply_flag and not self.prev_apply_flag:
            self.applied_force = np.random.uniform(self.wrench_min, self.wrench_max)
            self.applied_time = 0.0
            self.to_apply_time = np.random.uniform(0.1, 3.0)
        # print(self.applied_time, self.to_apply_time)
        # print("apply:",self.apply_flag)
        if self.apply_flag:
            force_to_apply = self.applied_force
            self.applied_time = self.applied_time + 1.0 / self.sim_freq
        else:
            force_to_apply = np.zeros((6, 1))
        self.prev_apply_flag = self.apply_flag
        if self.apply_flag and self.applied_time > self.to_apply_time:
            # print("reset!")
            self.reset()
        # print(self.apply_flag)
        # print(force_to_apply.flatten(), self.apply_flag)
        return np.copy(force_to_apply.flatten()), np.copy(self.apply_flag)

    def get_test_perturbation(self):
        return np.array([0, -22, 0, 0, 0, 0])
        # return np.array([-22,22,0,0,0,0])