import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

'''
---- DESCRIPTION ----

Differences from base version:
* 3 bonus states instead of 2.
* Different init position.
* Bonus reward is not given upon bonus collection. Instead, if the agent reaches the goal and it also previously collected the bonus, the final reward is doubled.
* If the agent did not reach the goal within the step limit, we give a reward proportional to the distance covered by the agent (x2 if the bonus was collected).
'''

class SparseCarBEnv(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.dt = 0.1
        self.tol = 0.01
        self.goal_state = 0.6
        self.bonus_states = [-0.5, 0.25, 0.8]
        self.seed()
        self._step = 0
        self.init_dist = -1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        self._step += 1
        u = np.clip(u, self.action_space.low, self.action_space.high)
        x = self.state[0] # car pos
        xd = self.state[1] # car vel
        b = self.state[2] # bonus pos
        gb = self.state[3] # got bonus?

        x_n = x + self.dt * xd
        xd_n = xd + self.dt * u

        done = False
        rwd = 0.
        dist_goal = np.abs(x - self.goal_state)
        dist_bonus = np.abs(x - b)
        if dist_bonus < self.tol and np.abs(xd) < self.tol:
            gb = 1 # flag: bonus was collected
        if dist_goal < self.tol and np.abs(xd) < self.tol:
            rwd += 1.
            if gb > 0: # if the goal was previously collected
                rwd *= 2.
            done = True

        if x_n < self.observation_space.low[0] or x_n > self.observation_space.high[0]:
            rwd -= 0.1 # penalty
            xd_n = 0. # stop the car

        # reward if the agent got closer to the goal, even if it didn't reach it and the episode is over
        if self._step >= self.spec.max_episode_steps and not done:
            rwd_partial = (1.0 - (dist_goal / self.init_dist))
            rwd += rwd_partial
            if gb > 0 and rwd_partial > 0: # double the rwd only if the agent got closer to the goal
                rwd *= 2.0
            done = True

        self.state[0] = x_n
        self.state[1] = xd_n
        self.state[2] = b
        self.state[3] = gb
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        rwd -= 0.001*np.sum(u**2)

        return self._get_obs(), rwd, done, {}

    def reset(self):
        self._step = 0
        self.state = np.array([0,0,0,0])
        self.state[2] = self.bonus_states[np.random.randint(0,len(self.bonus_states))]
        self.last_u = None
        self.init_dist = np.abs(self.state[0] - self.goal_state) # initial distance from the goal
        return self._get_obs()

    def _get_obs(self):
        return self.state
