import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

'''
---- DESCRIPTION ----
Multi-objective version of the LQR.
'''

class MOLqrEnv(gym.Env):

    def __init__(self, size, init_state, state_bound, eps=.1):
        self.init_state = init_state
        self.size = size  # dimensionality of state and action
        self.state_bound = state_bound
        self.action_space = spaces.Box(low=-state_bound, high=state_bound, shape=(size,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=(size,), dtype=np.float32)
        self.n_rewards = size
        self._seed()

        self.eps = eps
        if size == 2:
            self.utopia = -40*np.ones((1,size)) # 43.5379
            self.antiutopia = -95*np.ones((1,size)) # 90.6736
        elif size == 3:
            self.utopia = -50*np.ones((1,size)) # 54.0788
            self.antiutopia = -105*np.ones((1,size)) # 101.2145
        else:
            NotImplementedError('current implementation is limited to 2 and 3 objectives')


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        costs = np.zeros(self.size)
        for i in range(self.size):
            Q = self.eps * np.ones(self.size)
            R = (1. - self.eps) * np.ones(self.size)
            Q[i] = 1. - self.eps
            R[i] = self.eps
            costs[i] = np.sum(u**2*R) + np.sum(self.state**2*Q)
        self.state = np.clip(self.state + u, self.observation_space.low, self.observation_space.high)
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = self.init_state*np.ones((self.size,))
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state




    def riccati_matrix(self, K, gamma=1.):
        tolerance = 0.0001
        maxitr = 500
        I = np.eye(self.size)
        P_list = []
        for i in range(self.size):
            Q = self.eps * np.ones(self.size)
            R = (1. - self.eps) * np.ones(self.size)
            Q[i] = 1. - self.eps
            R[i] = self.eps
            P = I
            Pnew = np.diag(Q) + gamma*P + gamma*np.dot(K.T,P) + gamma*np.dot(P,K) + gamma*np.dot(K.T,P).dot(K) + np.dot(np.dot(K.T,np.diag(R)),K)
            itr = 0
            converged = False
            while not converged and itr < maxitr:
                P = Pnew
                Pnew = np.diag(Q) + gamma*P + gamma*np.dot(K.T,P) + gamma*np.dot(P,K) + gamma*np.dot(K.T,P).dot(K) + np.dot(np.dot(K.T,np.diag(R)),K)
                P_diff = P - Pnew
                if np.any(np.isnan(P_diff)) or np.any(np.isinf(P_diff)):
                    break
                converged = np.max(P_diff) < tolerance
                itr += 1
            P_list.append(P)
        return P_list

    def v_function(self, K, state, gamma=1.):
        V_list = []
        for P in riccati_matrix(K,gamma):
            V_list.append(-np.sum(np.dot(np.square(state),P), axis=1))
        return V_list

    def q_function(self, K, state, action, gamma=1.):
        I = np.eye(self.size)
        tmp = state + action
        Q_list = []
        for P, i in zip(self.riccati_matrix(K,gamma), range(self.size)):
            Q = self.eps * np.ones(self.size)
            R = (1. - self.eps) * np.ones(self.size)
            Q[i] = 1. - self.eps
            R[i] = self.eps
            Q_list.append(-np.sum(np.square(state)*Q + np.square(action)*R, axis=1, keepdims=True) - gamma*np.dot(np.square(tmp),P))
        return Q_list

    def avg_return(self, K, gamma=1.):
        J_list = []
        high = self.init_state*np.ones((self.size,))
        Sigma_s = np.diag(high+high)**2 / 12.
        for P in self.riccati_matrix(K,gamma):
            J_list.append(-np.trace(Sigma_s*P))
        return J_list
