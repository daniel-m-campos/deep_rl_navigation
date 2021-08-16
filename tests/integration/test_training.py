import gym
import pytest
import torch

from deep_rl_navigation import agent, training


def test_training():
    agent.DEVICE = torch.device("cpu")
    dqn_agent = agent.DQNAgent(state_size=8, action_size=4, seed=0)
    env = gym.make("LunarLander-v2")
    env.seed(0)
    scores = training.train(dqn_agent, env, n_episodes=1)
    assert len(scores) == 1
    assert scores[0] == pytest.approx(-327.87563499560366)
