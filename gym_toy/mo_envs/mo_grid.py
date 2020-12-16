import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

'''
---- DESCRIPTION ----
Simple NxN gridworld with rewards at the bottom row.
The agent starts in [0,0].
The closest reward (straight below the init pos) is the lowest.
Further rewards have higher values.
The frontier can be either convex or concave.

Observations and reward are normalized in [-1,1].
'''

class MOGridEnv(gym.Env):

    def __init__(self, n=5, convex=True):
        # Notice that size will be (n+1).
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.n = n
        self.n_rewards = 2

        self.rewards = np.zeros((n,n))
        if convex:
            self.rewards[n-1,:] = np.sqrt(n**2 - np.arange(n)**2)[::-1]
        else:
            self.rewards[n-1,:] = np.exp(np.log(n) - np.log(np.arange(n)+1))[::-1]
            # alpha = 0.9 # 0.99
            # self.rewards[n-1,:] = (alpha**np.arange(n)**2)[::-1]

        self.front = np.zeros((n,2))
        self.front[:,0] = self.rewards[n-1,:]
        self.front[:,1] = - (np.arange(n) + n) + 1
        self.utopia = self.front.max(axis=0) + 0.1
        self.antiutopia = self.front.min(axis=0) - 0.1

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.array([0, 0])
        self.last_u = None
        return self._get_obs()

    def step(self, action):
        if action == 0:
            if self.state[1] > 0:
                self.state[1] -= 1
        elif action == 1:
            if self.state[0] > 0:
                self.state[0] -= 1
        elif action == 2:
            if self.state[1] < self.n:
                self.state[1] += 1
        elif action == 3:
            if self.state[0] < self.n:
                self.state[0] += 1
        else:
            raise Exception('unexpected action: ' + str(action))

        rewards = np.zeros(2)
        rewards[0] = self.rewards[self.state[0],self.state[1]]
        rewards[1] = -1

        done = rewards[0] > 0

        return self._get_obs(), rewards, done, {}

    def _get_obs(self):
        return self.state / (self.n / 2.) - 1.
