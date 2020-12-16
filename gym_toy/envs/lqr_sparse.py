import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

'''
---- DESCRIPTION ----

Like LQR, but the reward is
r = - s'Qs - a'Ru        if ||s|| < 1
r =  - 1   - a'Ru        otherwise

Action is also bounded in [-1,1].
'''

class LqrSparseEnv(gym.Env):

    def __init__(self, size, init_state, state_bound):
        self.init_state = init_state
        self.size = size # dimensionality of state and action
        self.action_space = spaces.Box(low=-state_bound, high=state_bound, shape=(size,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=(size,), dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        if np.sqrt(np.sum(self.state**2)) < 1:
            costs = np.sum(u**2) + np.sum(self.state**2)
        else:
            costs = 1 + np.sum(u**2)
        self.state = np.clip(self.state + u, self.observation_space.low, self.observation_space.high)
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = self.init_state*np.ones((self.size,))
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state
