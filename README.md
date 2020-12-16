### Installation
```
pip3 install -e .
```

### Usage
```
import gym
import gym_toy
```

### Brief description of the environments

# Tabular

* `random_switch.py`        : 2D gridworld where the agent has to press a switch (5th action) in certain cells to get a reward.

# Continuous state / discrete action

* `gridworld.py`            : 2D env with sparse rewards.
* `dig.py`                  : 2D env where the agent has to dig land (5th action) to find rewards.

# Continuous control

* `gridworld_continuous.py` : 2D env sparse rewards.
* `lqr.py`                  : linear-quadratic regulator.
* `lqr_sparse.py`           : state penalty is always -1, except when the agent is close to the goal (distance < 1).
* `sparse_car.py`           : car moving on a 1D plane with sparse reward.
* `sparse_navi.py`          : agent navigating on a 2D env, with linear dynamics and sparse reward.
* `pendulum_sparse.py`      : like gym Pendulum-v0, but reward is sparse.

# Multi-objective

* `mo_lqr.py`               : multi-objective linear-quadratic regulator.
* `mo_grid.py`              : multi-objective gridworld (the farther the reward, the higher its value).
