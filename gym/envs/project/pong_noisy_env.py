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

class PongNoisyEnv(AtariEnv):
    """
    Adds Gaussian noise element-wise
    to the output state, with 
    mu = 0, sigma = sqrt(abs(state)).
    """
    def __init__(self, sigma=1.0, **kwargs):
        self.sigma = sigma
        super().__init__(**kwargs)
        assert isinstance(self.observation_space, spaces.Box)

    def _step(self, action):
        state, reward, done, info = super()._step(action)
        noise = np.random.normal(loc=0.0, scale=self.sigma, size=state.shape)
        noisy_state = state + np.sqrt(np.abs(state)) * noise

        # Ensure state doesn't violate bounds of observation space
        noisy_state = np.fmax(noisy_state, self.observation_space.low)
        noisy_state = np.fmin(noisy_state, self.observation_space.high)

        return noisy_state, reward, done, info