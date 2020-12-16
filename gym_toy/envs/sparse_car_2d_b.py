import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

'''
---- DESCRIPTION ----

Like sparse_car_b.py but in 2D.
'''

class SparseCar2DBEnv(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.dt = 0.1
        self.tol = 0.01
        self.goal_state = [0.6, 0]
        self.bonus_states = [[-0.5, 0.1],
                             [0.25, 0],
                             [0.8, -0.1]]
        self.seed()
        self._step = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        self._step += 1
        u = np.clip(u, self.action_space.low, self.action_space.high)
        x = self.state[0] # pos x
        y = self.state[1] # pos y
        xd = self.state[2] # vel x
        yd = self.state[3] # vel y
        bx = self.state[4] # bonus pos x
        by = self.state[5] # bonus pos y
        gb = self.state[6] # got bonus?

        x_n = x + xd * self.dt
        y_n = y + yd * self.dt
        xd_n = xd + u[0] * self.dt
        yd_n = yd + u[1] * self.dt

        done = False
        rwd = 0.0
        dist_goal = np.sqrt((x - self.goal_state[0])**2 + (y - self.goal_state[1])**2)
        dist_bonus = np.sqrt((x - bx)**2 + (y - by)**2)
        if dist_bonus < self.tol and np.abs(xd) < self.tol and np.abs(yd) < self.tol:
            gb = 1 # flag: bonus was collected
        if dist_goal < self.tol and np.abs(xd) < self.tol and np.abs(yd) < self.tol:
            rwd += 1.
            if gb > 0: # if the goal was previously collected
                rwd *= 2.
            done = True

        if x_n < self.observation_space.low[0] or x_n > self.observation_space.high[0]:
            rwd -= 0.1 # penalty
            xd_n = 0. # stop the car

        if y_n < self.observation_space.low[1] or y_n > self.observation_space.high[1]:
            rwd -= 0.1 # penalty
            yd_n = 0. # stop the car

        # reward if the agent got closer to the goal, even if it didn't reach it and the episode is over
        if self._step >= self.spec.max_episode_steps and not done:
            rwd_partial = (1.0 - (dist_goal / self.init_dist))
            rwd += rwd_partial
            if gb > 0 and rwd_partial > 0: # double the rwd only if the agent got closer to the goal
                rwd *= 2.0
            done = True

        self.state[0] = x_n
        self.state[1] = y_n
        self.state[2] = xd_n
        self.state[3] = yd_n
        self.state[6] = gb
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        rwd -= 0.001*np.sum(u**2)

        return self.state, rwd, done, {}

    def reset(self):
        self._step = 0
        b = np.random.randint(0,len(self.goal_state))
        self.state = np.array([0, # pos x
                               0, # pos y
                               0, # vel x
                               0, # vel y
                               self.bonus_states[b][0], # bonus pos x
                               self.bonus_states[b][1], # bonus pos y
                               0, # got bonus?
                               ])
        self.last_u = None
        self.init_dist = np.sqrt((self.state[0] - self.goal_state[0])**2 + (self.state[1] - self.goal_state[1])**2)
        return self._get_obs()

    def _get_obs(self):
        return self.state
