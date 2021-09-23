import pathlib

import gym
import torch

from deep_rl import agent_io, agent

RESOURCE_DIR = pathlib.Path(__file__).parent.parent / "resources"


def test_ddpg_agent_with_pendulum():
    agent.DEVICE = torch.device("cpu")
    ddpg_agent = agent.DDPGAgent(
        state_size=3,
        action_size=1,
        seed=2,
        fc1_units=400,
        fc2_units=300,
        use_batch_norm=False,
    )
    agent_io.load(ddpg_agent, (RESOURCE_DIR / "actor.pth", RESOURCE_DIR / "critic.pth"))
    env = gym.make("Pendulum-v0")
    env.seed(2)

    state = env.reset()
    for t in range(200):
        action = ddpg_agent.act(state, add_noise=False)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break

    env.close()
