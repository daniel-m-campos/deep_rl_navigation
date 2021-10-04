import abc
from typing import Tuple, Dict, Union, List, Any

import gym
import numpy as np
from unityagents import UnityEnvironment


class Environment(abc.ABC):
    @abc.abstractmethod
    def step(
        self, action
    ) -> Tuple[np.array, Union[float, List[float]], Union[bool, List[bool]], Dict]:
        pass

    @abc.abstractmethod
    def reset(self) -> np.array:
        pass

    @property
    @abc.abstractmethod
    def action_space(self) -> gym.Space:
        pass

    @property
    @abc.abstractmethod
    def observation_space(self) -> gym.Space:
        pass


class Navigation(Environment):
    default_path = "/usr/local/sbin/Banana.x86_64"

    def __init__(self, unity_env: UnityEnvironment, train_mode=True):
        super().__init__()
        self.train_mode = train_mode
        self._unity_env = unity_env
        self._brain_name = unity_env.brain_names[0]
        brain = unity_env.brains[unity_env.brain_names[0]]
        self._info = self._unity_env.reset(train_mode=train_mode)[self._brain_name]
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
    def from_binary_path(cls, binary_path: str = None, **kwargs):
        binary_path = cls.default_path if binary_path is None else binary_path
        return cls(UnityEnvironment(file_name=binary_path), **kwargs)


class ContinuousControl(Environment):
    default_path = "/usr/local/sbin/Reacher.x86_64"

    def __init__(self, unity_env: UnityEnvironment, train_mode=True):
        super().__init__()
        self.train_mode = train_mode
        self._unity_env = unity_env
        self._brain_name = unity_env.brain_names[0]
        brain = unity_env.brains[unity_env.brain_names[0]]
        self._info = self._unity_env.reset(train_mode=train_mode)[self._brain_name]
        self._action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(brain.vector_action_space_size,),
            dtype=np.float32,
        )
        self._observation_space = gym.spaces.Box(
            low=-100,
            high=100,
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

    def reset(self) -> np.array:
        self._info = self._unity_env.reset(train_mode=self.train_mode)[self._brain_name]
        return self._info.vector_observations[0]

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @classmethod
    def from_binary_path(cls, binary_path: str = None, **kwargs):
        binary_path = cls.default_path if binary_path is None else binary_path
        return cls(UnityEnvironment(file_name=binary_path), **kwargs)


class Tennis(Environment):
    default_path = "/usr/local/sbin/Tennis.x86_64"

    def __init__(self, unity_env: UnityEnvironment, train_mode=True):
        super().__init__()
        self.train_mode = train_mode
        self._unity_env = unity_env
        self._brain_name = unity_env.brain_names[0]
        brain = unity_env.brains[unity_env.brain_names[0]]
        self._info = self._unity_env.reset(train_mode=train_mode)[self._brain_name]
        self._action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(brain.vector_action_space_size,),
            dtype=np.float32,
        )
        self._observation_space = gym.spaces.Box(
            low=-100,
            high=100,
            shape=(len(self._info.vector_observations[0]),),
            dtype=np.float32,
        )

    def step(self, action) -> Tuple[np.array, Any, Any, Dict]:
        self._info = self._unity_env.step(action)[self._brain_name]
        return (
            self._info.vector_observations,
            self._info.rewards,
            self._info.local_done,
            {},
        )

    def reset(self) -> np.array:
        self._info = self._unity_env.reset(train_mode=self.train_mode)[self._brain_name]
        return self._info.vector_observations

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @classmethod
    def from_binary_path(cls, binary_path: str = None, **kwargs):
        binary_path = cls.default_path if binary_path is None else binary_path
        return cls(UnityEnvironment(file_name=binary_path), **kwargs)


def create(env_name: str, binary_path: str = None, **kwargs) -> Environment:
    env_name = env_name.replace("_", " ").title().replace(" ", "")
    if env_name in Navigation.__name__:
        env = Navigation
    elif env_name in ContinuousControl.__name__:
        env = ContinuousControl
    elif env_name in Tennis.__name__:
        env = Tennis
    else:
        raise NotImplementedError(f"{env_name} not supported")

    return env.from_binary_path(binary_path, **kwargs)
