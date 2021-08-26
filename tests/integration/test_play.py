import pytest
import torch

from deep_rl import agent, environment, play


@pytest.fixture
def navigation_env():
    return environment.NavigationEnv.from_unity_binary()


def test_play(navigation_env: environment.NavigationEnv):
    agent.DEVICE = torch.device("cpu")
    dqn_agent = agent.DQNAgent(
        state_size=navigation_env.observation_space.shape[0],
        action_size=navigation_env.action_space.n,
        seed=0,
    )
    score = play.play(dqn_agent, navigation_env, max_steps=100)
    print(f"Score: {score}")
