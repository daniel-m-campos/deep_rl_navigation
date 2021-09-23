import pathlib

from deep_rl import agent_io
from deep_rl.agent import DQNAgent, DDPGAgent

RESOURCE_DIR = pathlib.Path(__file__).parent.parent / "resources"


def test_load_dqn_agent():
    agent = DQNAgent(state_size=8, action_size=4, seed=0)
    agent_io.load(agent, RESOURCE_DIR / "checkpoint.pth")


def test_load_ddpg_agent():
    agent = DDPGAgent(
        state_size=3,
        action_size=1,
        seed=2,
        fc1_units=400,
        fc2_units=300,
        use_batch_norm=False,
    )
    filenames = (RESOURCE_DIR / "actor.pth", RESOURCE_DIR / "critic.pth")
    agent_io.load(agent, filenames)
