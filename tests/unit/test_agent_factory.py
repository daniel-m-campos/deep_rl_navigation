import pathlib

import pytest

from deep_rl import agent, agent_factory

RESOURCE_DIR = pathlib.Path(__file__).parent.parent / "resources"


def test_create_raises_not_implemented_error():
    with pytest.raises(NotImplementedError):
        _ = agent_factory.create("Doesnt exist", "test.pth")


def test_create_dqn_agent():
    dqn_agent = agent_factory.create(
        "dqn",
        RESOURCE_DIR / "checkpoint.pth",
        state_size=8,
        action_size=4,
        seed=0,
    )
    assert isinstance(dqn_agent, agent.DQNAgent)


def test_create_ddpg_agent():
    dqn_agent = agent_factory.create(
        "ddpg",
        (RESOURCE_DIR / "actor.pth", RESOURCE_DIR / "critic.pth"),
        state_size=3,
        action_size=1,
        seed=2,
        fc1_units=400,
        fc2_units=300,
        use_batch_norm=False,
    )
    assert isinstance(dqn_agent, agent.DDPGAgent)
