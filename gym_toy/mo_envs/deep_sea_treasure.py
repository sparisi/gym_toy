import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from matplotlib import pyplot as plt

'''
---- DESCRIPTION ----
Modified version with convex frontier, as in https://arxiv.org/abs/1908.08342

Observations and reward are normalized in [-1,1].
'''

class DSTEnv(gym.Env):

    def __init__(self, normalize_reward=True):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1., shape=(2,), dtype=np.float32)
        self.n_rewards = 2
        self.n = 10
        self.normalize_reward = normalize_reward

        self.rewards = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, 8.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, -10, 11.5, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, -10, -10, 14.0, 15.1, 16.1, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 19.6, 20.3, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 22.4, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, -10, 23.7, 0]]
        )
        self.front = np.array(
            [[0.7, -1],
             [8.2, -3],
             [11.5, -5],
             [14.0, -7],
             [15.1, -8],
             [16.1, -9],
             [19.6, -13],
             [20.3, -14],
             [22.4, -17],
             [23.7, -19]]
        )
        self.utopia = self.front.max(axis=0) + 0.1
        self.antiutopia = self.front.min(axis=0) - 0.1

        # for rendering
        self._fig = None
        self._axs = None
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
        state_copy = np.copy(self.state)
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

        if self.rewards[self.state[0],self.state[1]] < 0:
            self.state = state_copy

        rewards = np.zeros(2)
        rewards[0] = self.rewards[self.state[0],self.state[1]]
        rewards[1] = -1

        done = rewards[0] > 0

        if self.normalize_reward:
            rewards[0] /= np.max(np.abs(self.rewards))

        return self._get_obs(), rewards, done, {}

    def _get_obs(self):
        return self.state / (self.n / 2.) - 1.

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.get_all_land()
        elif mode == 'human':
            if self._fig is None:
                plt.ion()
                self._fig, self._axs = plt.subplots(1, figsize=(3,4))
                self._fig.tight_layout(pad=2.0)
                plt.contourf(np.flipud(self.rewards))
                self._agent_dot = self._axs.plot(0, 0, 'o', color='blue')
                self._fig.suptitle('DeepSeaTreasure-v0')
            self._agent_dot[0].set_data(self.state[1],
                            self.n-self.state[0]) # convert cartesian to matrix coord

            self._fig.canvas.flush_events()
            plt.show(block=False)
            # plt.pause(0.0000001)
        else:
            raise Exception('unexpected render mode: ' + mode)
