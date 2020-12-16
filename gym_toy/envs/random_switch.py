import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

'''
---- DESCRIPTION ----

Gridworld where random cells contains switches.
Switching some of them on gives a reward.
'''

class RandSwitchEnv(gym.Env):

    def __init__(self, n=20):
        self.n = n
        self.observation_space = spaces.Box(low=np.array([0,0,-2]), high=np.array([n-1,n-1,3]), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.og_grid = None # original grid (at reset time)
        self.grid = None # current grid
        self.goal = None # id of switches giving reward

        # Default env
        self.gen_grid(0)
        self.set_goal(1)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, init_state=None):
        self.grid = self.og_grid.copy()

        choices = []
        for c in [[0, 0], [0, self.n-1], [self.n-1, 0], [self.n-1, self.n-1], [self.n//2, self.n//2]]:
            if self.grid[c[0], c[1]] != -1:
                choices.append(c)
        self.state = np.array(choices[np.random.choice(len(choices))])

        # self.state = np.random.randint(0, self.n, (2,))
        # while self.grid[self.state[0], self.state[1]] != 0:
        #     self.state = np.random.randint(0, self.n, (2,))
        self.last_u = None
        return self._get_obs()

    def step(self, action):
        reward = 0
        done = False

        if action == 0:
            if self.state[1] > 0 and self.grid[self.state[0], self.state[1] - 1] != -1:
                self.state[1] -= 1
        elif action == 1:
            if self.state[0] > 0 and self.grid[self.state[0] - 1, self.state[1]] != -1:
                self.state[0] -= 1
        elif action == 2:
            if self.state[1] < self.n - 1 and self.grid[self.state[0], self.state[1] + 1] != -1:
                self.state[1] += 1
        elif action == 3:
            if self.state[0] < self.n - 1 and self.grid[self.state[0] + 1, self.state[1]] != -1:
                self.state[0] += 1
        elif action == 4:
            if self.grid[self.state[0], self.state[1]] == self.goal:
                reward = 1
            if self.grid[self.state[0], self.state[1]] > 0:
                # done = True
                self.grid[self.state[0], self.state[1]] = -2 # -2 is for pressed switch, which cannot be switched back
        else:
            raise Exception('unexpected action: ' + str(action))

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # return self.state
        return np.append(self.state, self.grid[self.state[0], self.state[1]] > 0)

    def set_goal(self, goal):
        self.goal = goal

    def gen_grid(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.grid = np.zeros((self.n,self.n))

        # randomly generate a grid
        for i in range(self.n):
            for j in range(self.n):
                r = np.random.rand()
                if r < 0.05:
                    self.grid[i,j] = 1
                elif r < 0.1:
                    self.grid[i,j] = 2
                elif r < 0.15:
                    self.grid[i,j] = 3
                elif r < 0.4:
                    self.grid[i,j] = -1 # can't go there
                else:
                    self.grid[i,j] = 0 # empty

        # check for isolated cells and make them inaccessible
        for i in range(self.n):
            for j in range(self.n):
                if (i == self.n - 1 or self.grid[i+1,j] == 0) and \
                   (i == 0          or self.grid[i-1,j] == 0) and \
                   (j == self.n - 1 or self.grid[i,j+1] == 0) and \
                   (j == 0          or self.grid[i,j-1] == 0):
                   self.grid[i,j] = 0

        self.og_grid = self.grid.copy()
