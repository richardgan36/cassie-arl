import gymnasium as gym
import numpy as np


class CassieEnv(gym.Env):
    def __init__(self):
        super(CassieEnv, self).__init__()
        
        # Cassie has 10 actuated joints (5 per leg)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        
        # Observation can include qpos, qvel, pelvis orientation, IMU, etc.
        # For simplicity, let's assume: 20 joint states + 20 velocities + 7 pelvis/IMU = 47
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32
        )
        
        self.state = np.zeros(self.observation_space.shape)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start from default pose
        self.state = np.zeros(self.observation_space.shape)
        return self.state, {}

    def step(self, action):
        # Clip action to action space
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Placeholder: apply action to simulation (here we just add some noise)
        # In a real sim, this is where you would call mujoco_py or mjx functions
        self.state += 0.01 * action.repeat(4)[:self.state.shape[0]]
        
        # Placeholder reward: encourage small actions (for testing)
        reward = -np.sum(action**2)
        
        # Placeholder done: never done
        done = False
        
        # Info dict (can include diagnostics)
        info = {}
        
        return self.state, reward, done, False, info  # Gymnasium API: (obs, reward, terminated, truncated, info)

    def render(self, mode='human'):
        pass  # Optional visualization

    def close(self):
        pass
