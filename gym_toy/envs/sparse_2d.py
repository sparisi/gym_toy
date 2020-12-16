import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from matplotlib import pyplot as plt

'''
---- DESCRIPTION ----

2D environment with continuous state and discrete action.
There are multiple goals, with the furthest yielding the highest reward.
The initial position is fixed.
The episode may or may not end when a reward is collected.
Transition has Gaussian noise.

With the default implementation, the highest reward is located in the top-right
corner and needs 40 steps (on avg) to be collected.

Observations and reward are normalized in [-1,1] and [0,1], respectively.
'''


class Sparse2DEnv(gym.Env):

    def __init__(self, inf_hor=False):
        self.inf_hor = inf_hor
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)

        # for rendering
        self._fig = None
        self._axs = None
        self._rwd_dots = None
        self._agent_dot = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        dist = np.sqrt(np.sum((self.state - self.rwd_states) ** 2, 1))
        is_close = np.where(dist <= self.rwd_radius)[0]
        done = False
        rwd = 0.
        if is_close.size > 0:
            rwd = self.rwd_magnitude[is_close[0]]
            done = not self.inf_hor

        if action == 0:
            self.state[0] -= 1. / self.scale
        elif action == 1:
            self.state[0] += 1. / self.scale
        elif action == 2:
            self.state[1] -= 1. / self.scale
        elif action == 3:
            self.state[1] += 1. / self.scale
        else:
            raise Exception('unexpected action: ' + str(action))

        self.state += np.random.normal(scale=self.noise_std, size=2) / self.scale
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        return self._get_obs(), rwd, done, {}

    def reset(self):
        # self.state = self.np_random.uniform(low=self.observation_space.low, high=self.observation_space.high)
        self.state = np.array([0.,0.])
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state


    def render(self, mode='human'):
        if mode == 'rgb_array':
            raise NotImplementedError
        elif mode == 'human':
            if self._fig is None:
                plt.ion()
                self._fig, self._axs = plt.subplots(1, figsize=(3,4))
                self._fig.tight_layout(pad=2.0)
                self._axs.set_xlim(self.observation_space.low[0], \
                                self.observation_space.high[0])
                self._axs.set_ylim(self.observation_space.low[1], \
                                self.observation_space.high[1])
                self._rwd_dots = self._axs.plot(self.rwd_states[:,0], self.rwd_states[:,1], 'o', color='red')
                self._agent_dot = self._axs.plot(0, 0, 'o', color='blue')
                self._fig.suptitle('GridWorld-v0')
            self._agent_dot[0].set_data(self.state[0], self.state[1])

            self._fig.canvas.flush_events()
            plt.show(block=False)
            # plt.pause(0.0000001)
        else:
            raise Exception('unexpected render mode: ' + mode)




class Sparse2DSmallEnv(Sparse2DEnv):
    def __init__(self, inf_hor=False):
        super().__init__(inf_hor)
        self.scale = 5.
        self.noise_std = 0.05
        # the reward is collected if the distance of the agent from the goal is within this radius
        self.rwd_radius = 1. / self.scale
        self.rwd_states = np.array([[2., 2.], [-2, 3], [4, 4]]) / self.scale
        self.rwd_magnitude = np.array([1., 4, 10])
        self.rwd_magnitude /= np.max(self.rwd_magnitude)



class Sparse2DBigEnv(Sparse2DEnv):
    def __init__(self, inf_hor=False):
        super().__init__(inf_hor)
        self.scale = 20.
        self.noise_std = 0.05
        self.rwd_radius = 1. / self.scale
        self.rwd_states = np.array([[2., 2.], [-2, 3], [10, -2], [19, 19]]) / self.scale
        self.rwd_magnitude = np.array([1., 4, 5, 10])
        self.rwd_magnitude /= np.max(self.rwd_magnitude)
