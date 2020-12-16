import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

'''
---- DESCRIPTION ----

Car moving on a line.
* The state is the agent position (x) and velocity (xd), and the bonus location (b).
* The action is the acceleration.
* The location of the bonus is randomly chosen at the beginning of the episode among
  two possible positions: either behind the initial position of the agent, or behind the goal.
* The initial position is fixed, as well as the goal position.
* The agent gets the bonus if it is very close to it with almost 0 velocity. The bonus then is moved to the goal state.
* The episode ends when the agent is very close to the goal with almost 0 velocity.
* The reward for the bonus is given only at the end of the episode if the agent reaches the goal.
* If the agent hits the position bounds, it gets a penalty and its velocity is reset to 0.
* Additionally, a small squared penalty on the action is given.

To recap, the reward is given only:
* At the goal (+1, or +2 if the goal was collected),
* At the boundaries (-0.1),
* Always squared penalty on the action.
'''

class SparseCarEnv(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.dt = 0.1
        self.tol = 0.01
        self.goal_state = 0.8
        self.bonus_states = [-0.2, 1.]
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        u = np.clip(u, self.action_space.low, self.action_space.high)
        x = self.state[0]
        xd = self.state[1]
        b = self.state[2]

        x_n = x + self.dt * xd
        xd_n = xd + self.dt * u

        done = False
        rwd = 0.
        dist_goal = np.abs(x - self.goal_state)
        dist_bonus = np.abs(x - b)
        if dist_bonus < self.tol and np.abs(xd) < self.tol:
            b = self.goal_state # place bonus where the goal is
        if dist_goal < self.tol and np.abs(xd) < self.tol:
            rwd += 1.
            if dist_bonus < self.tol and np.abs(xd) < self.tol: # if the goal was previously collected
                rwd += 1.
            done = True

        if x_n < self.observation_space.low[0] or x_n > self.observation_space.high[0]:
            rwd -= 0.1 # penalty
            xd_n = 0. # stop the car (hit state boundaries)

        self.state[0] = x_n
        self.state[1] = xd_n
        self.state[2] = b
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        rwd -= 0.001*np.sum(u**2)

        return self._get_obs(), rwd, done, {}

    def reset(self):
        self.state = np.array([0,0,0])
        self.state[2] = self.bonus_states[np.random.randint(0,len(self.bonus_states))]
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state
