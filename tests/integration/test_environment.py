import gym
import numpy as np
import pytest

from deep_rl_navigation import environment, UNITY_BINARY


@pytest.fixture(scope="class")
def navigation_env():
    return environment.NavigationEnv(
        environment.UnityEnvironment(file_name=UNITY_BINARY)
    )


@pytest.fixture(scope="class")
def initial_state():
    return np.array(
        [
            0.0,
            1.0,
            0.0,
            0.0,
            0.16895212,
            0.0,
            1.0,
            0.0,
            0.0,
            0.20073597,
            1.0,
            0.0,
            0.0,
            0.0,
            0.12865657,
            0.0,
            1.0,
            0.0,
            0.0,
            0.14938059,
            1.0,
            0.0,
            0.0,
            0.0,
            0.58185619,
            0.0,
            1.0,
            0.0,
            0.0,
            0.16089135,
            0.0,
            1.0,
            0.0,
            0.0,
            0.31775284,
            0.0,
            0.0,
        ]
    )


class TestNavigationEnv:
    # https://docs.pytest.org/en/latest/how-to/fixtures.html#teardown-cleanup-aka-fixture-finalization
    def test_reset(self, navigation_env, initial_state):
        state = navigation_env.reset()
        assert np.allclose(state, initial_state)

    def test_action_space(self, navigation_env):
        action_space = navigation_env.action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        assert action_space.n == 4
