import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy.stats as stats

from matplotlib import pyplot as plt

'''
---- DESCRIPTION ----

2D environment with continuous state and discrete action.

There are multiple sparse rewards which have to be dug.
Observations are the (x,y) position of the agent, and the amount of land in (x,y).

The land has a Gaussian distribution centered on the rewards, i.e.,
   land = sum_i N(rwd_pos_i, land_std)

The agent can move or dig.
The land is dug following another Gaussian, with a smaller std, i.e.,
   land_dig = N(agent_pos, dig_std) * dig_amount

The value dig_amount is fixed (default code requires 10 digs to find the rewards).
The land given by the Gaussians is normalized to be in [0,1].

The amount of land at any point is in [0,1] (cannot be dug below 0).

The episode may or may not end when a reward is collected.
'''

# TODO: if found rwd disappear, so the agent has to move to the next one
# TODO: different std for each rwd
# TODO: land prevents walking, must be dug first

class DigEnv(gym.Env):

    def __init__(self, inf_hor=True):
        self.inf_hor = inf_hor
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.array([1.,1.,0]), high=np.array([1.,1.,1.]), dtype=np.float32)
        self.scale = 20.
        self.rwd_radius = 1. / self.scale # the reward is collected if the distance of the agent from the goal is within this radius

        self.land_std = 5. / self.scale ** 2 # 'spread' of land

        self.rwd_land = [] # list of Gaussian for rwd land
        self.rwd_states = np.array([[-7, -15], [-5, 3], [10, -2], [20, 20]]) / self.scale # default rwds
        for rwd in self.rwd_states:
            self.rwd_land.append(stats.multivariate_normal(rwd, self.land_std))

        self.dig_land = [] # list of Gaussian for dug land
        self.dig_depth = [] # list of amount of land dug at every step
        self.dig_amount = 0.1

        # for normalization
        self.land_normal = self.rwd_land[-1].pdf(self.rwd_states[-1])

        # for rendering
        low = self.observation_space.low[:2]
        high = self.observation_space.high[:2]
        X = np.linspace(low[0], high[0], 20)
        Y = np.linspace(low[1], high[1], 20)
        self._X, self._Y = np.meshgrid(X, Y)
        self._XY = np.stack([self._X.flatten(), self._Y.flatten()], axis=-1)
        self._fig = None


    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.get_all_land()
        elif mode == 'human':
            if self._fig is None:
                plt.ion()
                self._fig, axs = plt.subplots(1, \
                                              subplot_kw=dict(projection='3d'), \
                                              figsize=(3,4))
                self._fig.tight_layout(pad=2.0)
            self._fig.suptitle('Dig-v0')
            axs = self._fig.axes
            Z = self.get_all_land().reshape(self._X.shape)
            axs[0].cla()
            axs[0].plot_surface(self._X, self._Y, Z, cmap='viridis', alpha=0.7)
            axs[0].plot(self.rwd_states[:,0], self.rwd_states[:,1], np.zeros(self.rwd_states.shape[0]), 'o', color='red')
            axs[0].plot(self.state[:,None][0,:], self.state[:,None][1,:], 'o', color='blue')
            axs[0].set_zlim(self.observation_space.low[2], \
                            self.observation_space.high[2])
            plt.pause(0.0001)
        else:
            raise Exception('unexpected render mode: ' + mode)


    def get_state_land(self):
        land = 0.
        for l in self.rwd_land:
            land += l.pdf(self.state) / self.land_normal
        for l, d in zip(self.dig_land, self.dig_depth):
            land -= l.pdf(self.state) / self.land_normal * d
        return np.clip(land, \
                       self.observation_space.low[2], \
                       self.observation_space.high[2])


    def get_all_land(self):
        land = 0.
        for l in self.rwd_land:
            land += l.pdf(self._XY) / self.land_normal
        for l, d in zip(self.dig_land, self.dig_depth):
            land -= l.pdf(self._XY) / self.land_normal * d
        return np.clip(land, \
                       self.observation_space.low[2], \
                       self.observation_space.high[2])


    def dig(self, u):
        self.dig_land.append(stats.multivariate_normal(self.state, self.land_std))
        self.dig_depth.append(u)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        done = False
        rwd = 0.

        dist = np.sqrt(np.sum((self.state - self.rwd_states) ** 2, 1))
        is_close = np.where(dist <= self.rwd_radius)[0]
        if is_close.size > 0 and self.get_state_land() <= 0:
            rwd = 1.
            done = not self.inf_hor

        if action == 0:
            self.state[0] -= 1. / self.scale
        elif action == 1:
            self.state[0] += 1. / self.scale
        elif action == 2:
            self.state[1] -= 1. / self.scale
        elif action == 3:
            self.state[1] += 1. / self.scale
        elif action == 4:
            self.dig(self.dig_amount)
        else:
            raise Exception('unexpected action: ' + str(action))

        # if action < 4:
            # self.state[:2] += np.random.normal(scale=0.1, size=2) / self.scale
        self.state = np.clip(self.state, \
                             self.observation_space.low[:2], \
                             self.observation_space.high[:2])
        return self._get_obs(), rwd, done, {}


    def randomize(self):
        self.rwd_states = (np.random.rand(4,2) - 0.5) * 2
        self.rwd_land = []
        for rwd in self.rwd_states:
            self.rwd_land.append(stats.multivariate_normal(rwd, self.land_std))


    def reset(self):
        self.state = self.np_random.uniform(low=self.observation_space.low[:2], \
                                            high=self.observation_space.high[:2])
        self.dig_land = []
        self.dig_depth = []
        self.last_u = None
        return self._get_obs()


    def _get_obs(self):
        return np.append(self.state, self.get_state_land())
