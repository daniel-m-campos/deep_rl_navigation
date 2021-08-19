import abc
from typing import Tuple, Dict

import gym
import numpy as np
from unityagents import UnityEnvironment

from deep_rl_navigation import UNITY_BINARY


class Environment(abc.ABC):
    @abc.abstractmethod
    def step(self, action) -> Tuple[np.array, float, bool, Dict]:
        pass

    @abc.abstractmethod
    def reset(self) -> np.array:
        pass

    @abc.abstractmethod
    def action_space(self) -> gym.Space:
        pass

    @abc.abstractmethod
    def observation_space(self) -> gym.Space:
        pass


class NavigationEnv(Environment):
    def __init__(self, unity_env: UnityEnvironment, train_mode=True):
        super().__init__()
        self.train_mode = train_mode
        self._unity_env = unity_env
        self._brain_name = unity_env.brain_names[0]
        brain = unity_env.brains[unity_env.brain_names[0]]
        self._info = self._unity_env.reset(train_mode=True)[self._brain_name]
        self._action_space = gym.spaces.Discrete(brain.vector_action_space_size)
        self._observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(self._info.vector_observations[0]),),
            dtype=np.float32,
        )

    def step(self, action) -> Tuple[np.array, float, bool, Dict]:
        self._info = self._unity_env.step(action)[self._brain_name]
        return (
            self._info.vector_observations[0],
            self._info.rewards[0],
            self._info.local_done[0],
            {},
        )

    def reset(self):
        self._info = self._unity_env.reset(train_mode=self.train_mode)[self._brain_name]
        return self._info.vector_observations[0]

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @classmethod
    def from_unity_binary(cls, **kwargs):
        return cls(UnityEnvironment(file_name=UNITY_BINARY), **kwargs)
