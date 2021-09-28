import gym
import numpy as np
import pytest

from deep_rl import environment


@pytest.fixture(scope="class")
def tennis_env():
    return environment.Tennis.from_binary_path()


@pytest.fixture(scope="class")
def initial_state():
    return np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -7.389936447143555,
            -1.5,
            -0.0,
            0.0,
            6.83172082901001,
            5.9960761070251465,
            -0.0,
            0.0,
        ]
    )


class TestTennisEnv:
    # https://docs.pytest.org/en/latest/how-to/fixtures.html#teardown-cleanup-aka-fixture-finalization

    def test_tennis_reset(self, tennis_env, initial_state):
        state = tennis_env.reset()
        assert np.allclose(
            state,
            initial_state,
        )

    def test_tennis_action_space(self, tennis_env):
        action_space = tennis_env.action_space
        assert isinstance(action_space, gym.spaces.Box)
        assert action_space.shape[0] == 2
