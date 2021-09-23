import gym
import pytest
import torch

from deep_rl import agent, train


def test_train_dqn_agent_with_lunar_lander():
    agent.DEVICE = torch.device("cpu")
    dqn_agent = agent.DQNAgent(state_size=8, action_size=4, seed=0)
    env = gym.make("LunarLander-v2")
    env.seed(0)
    scores = train.train(dqn_agent, env, n_episodes=1)
    assert len(scores) == 1
    assert scores[0] == pytest.approx(-327.87563499560366)


def test_train_ddpg_agent_with_pendulum():
    agent.DEVICE = torch.device("cpu")
    ddpg_agent = agent.DDPGAgent(
        state_size=3,
        action_size=1,
        seed=0,
        fc1_units=400,
        fc2_units=300,
        use_batch_norm=False,
    )
    env = gym.make("Pendulum-v0")
    env.seed(0)
    scores = train.train(ddpg_agent, env, n_episodes=2)
    assert len(scores) == 2
    assert sum(scores) / len(scores) == pytest.approx(-1356.5552154090365)
