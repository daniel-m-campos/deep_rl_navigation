from deep_rl_navigation import agent_io
from deep_rl_navigation.agent import DQNAgent


def test_load():
    agent = DQNAgent(state_size=8, action_size=4, seed=0)
    agent_io.load(agent, "../resources/checkpoint.pth")
