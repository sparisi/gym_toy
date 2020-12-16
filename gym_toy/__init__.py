from gym.envs.registration import register
import numpy as np

register(
    id='RandSwitch-v0',
    entry_point='gym_toy.envs:RandSwitchEnv',
    max_episode_steps=50,
)

register(
    id='SparseGrid-v0',
    entry_point='gym_toy.envs:SparseGridEnv',
    max_episode_steps=50,
)

register(
    id='Dig-v0',
    entry_point='gym_toy.envs:DigEnv',
    max_episode_steps=100,
)

register(
    id='Sparse2DSmall-v0',
    entry_point='gym_toy.envs:Sparse2DSmallEnv',
    max_episode_steps=10,
)

register(
    id='Sparse2DBig-v0',
    entry_point='gym_toy.envs:Sparse2DBigEnv',
    max_episode_steps=100,
)

register(
    id='PendulumSparse-v0',
    entry_point='gym_toy.envs:PendulumSparseEnv',
    max_episode_steps=200,
)

register(
    id='Lqr-v0',
    entry_point='gym_toy.envs:LqrEnv',
    max_episode_steps=150,
    kwargs={'size' : 2, 'init_state' : 10., 'state_bound' : np.inf},
)

register(
    id='LqrSparse-v0',
    entry_point='gym_toy.envs:LqrSparseEnv',
    max_episode_steps=150,
    kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : np.inf},
)

register(
    id='SparseCar-v0',
    entry_point='gym_toy.envs:SparseCarEnv',
    max_episode_steps=100,
)

register(
    id='SparseCarB-v0',
    entry_point='gym_toy.envs:SparseCarBEnv',
    max_episode_steps=100,
)

register(
    id='SparseCar2DB-v0',
    entry_point='gym_toy.envs:SparseCar2DBEnv',
    max_episode_steps=100,
)

# MO ------------------------

register(
    id='MOLqr-v0',
    entry_point='gym_toy.mo_envs:MOLqrEnv',
    max_episode_steps=150,
    kwargs={'size' : 2, 'init_state' : 10., 'state_bound' : np.inf},
)

register(
    id='MOGrid-v0',
    entry_point='gym_toy.mo_envs:MOGridEnv',
    max_episode_steps=30,
    kwargs={'n' : 5, 'convex' : True},
)

register(
    id='DeepSeaTreasure-v0',
    entry_point='gym_toy.mo_envs:DSTEnv',
    max_episode_steps=100,
)
