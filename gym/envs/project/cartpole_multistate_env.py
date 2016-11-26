import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.classic_control.mountain_car import MountainCarEnv
from gym.envs.atari.atari_env import AtariEnv
from gym import spaces
logger = logging.getLogger(__name__)

class CartPoleMultiStateEnv(CartPoleEnv):
    """
    Expands output state to include
    previous states.
    """
    def __init__(self, num_prev_states=2, **kwargs):
        self.state_history = []
        self.num_prev_states = num_prev_states
        super().__init__(**kwargs)

        # Expand observation space
        assert isinstance(self.observation_space, spaces.Box)
        all_lows = np.hstack([np.copy(self.observation_space.low) \
            for _ in range(self.num_prev_states)])
        all_highs = np.hstack([np.copy(self.observation_space.high) \
            for _ in range(self.num_prev_states)])
        self.observation_space = spaces.Box(all_lows, all_highs)

    def _step(self, action):
        state, reward, done, info = super()._step(action)
        self.state_history = self.state_history[1:] + [state]
        combined_state = np.hstack(self.state_history)
        return np.array(combined_state), reward, done, info

    def _reset(self):
        state = super()._reset()
        self.state_history = [np.copy(state) for _ in range(self.num_prev_states)]
        combined_state = np.hstack(self.state_history)
        return np.array(combined_state)