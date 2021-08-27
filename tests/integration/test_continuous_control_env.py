import gym
import numpy as np
import pytest

from deep_rl import environment


@pytest.fixture(scope="class")
def continuous_control_env():
    return environment.ContinuousControl.from_binary_path()


@pytest.fixture(scope="class")
def initial_state():
    return np.array(
        [
            0.0,
            -4.0,
            0.0,
            1.0,
            -0.0,
            -0.0,
            -4.371138828673793e-08,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -10.0,
            0.0,
            1.0,
            -0.0,
            -0.0,
            -4.371138828673793e-08,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -6.304084777832031,
            -1.0,
            -4.925292015075684,
            0.0,
            1.0,
            0.0,
            -0.5330140590667725,
        ]
    )


class TestContinuousControlEnv:
    # https://docs.pytest.org/en/latest/how-to/fixtures.html#teardown-cleanup-aka-fixture-finalization

    def test_continuous_control_reset(self, continuous_control_env, initial_state):
        state = continuous_control_env.reset()
        assert np.allclose(
            state,
            initial_state,
        )

    def test_continuous_control_action_space(self, continuous_control_env):
        action_space = continuous_control_env.action_space
        assert isinstance(action_space, gym.spaces.Box)
        assert action_space.shape[0] == 4
