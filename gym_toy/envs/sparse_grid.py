import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from matplotlib import pyplot as plt

'''
---- DESCRIPTION ----
Simple NxN gridworld with fixed initial position and one reward at the opposite
corner.

Observations and reward are normalized in [-1,1].
'''

class SparseGridEnv(gym.Env):

    def __init__(self, n=4):
        # Notice that size will be (n+1).
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.n = n

        # for rendering
        self._fig = None
        self._axs = None
        self._rwd_dots = None
        self._agent_dot = None

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.array([0, 0])
        self.last_u = None
        return self._get_obs()

    def step(self, action):
        reward = 0
        done = False
        if self.state[0] == self.n and self.state[1] == self.n:
            reward = 1
            done = True

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

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.state / (self.n / 2.) - 1.

    def render(self, mode='human'):
        if mode == 'rgb_array':
            raise NotImplementedError
        elif mode == 'human':
            if self._fig is None:
                plt.ion()
                self._fig, self._axs = plt.subplots(1, figsize=(3,4))
                self._fig.tight_layout(pad=2.0)
                self._axs.set_xlim(0, self.n)
                self._axs.set_ylim(0, self.n)
                self._rwd_dots = self._axs.plot(self.n, self.n, 'o', color='red')
                self._agent_dot = self._axs.plot(0, 0, 'o', color='blue')
                self._fig.suptitle('GridWorld-v0')
            self._agent_dot[0].set_data(self.state[0], self.state[1])

            self._fig.canvas.flush_events()
            plt.show(block=False)
            # plt.pause(0.0000001)
        else:
            raise Exception('unexpected render mode: ' + mode)
