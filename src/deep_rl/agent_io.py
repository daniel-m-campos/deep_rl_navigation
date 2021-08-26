import torch

from deep_rl.agent import Agent, DQNAgent


def save(agent: Agent, filename: str):
    # TODO: Add DDPGAgent support
    if isinstance(agent, DQNAgent):
        torch.save(agent.qnetwork_local.state_dict(), filename)
    else:
        raise NotImplementedError


def load(agent: Agent, filename: str):
    # TODO: Add DDPGAgent support
    if isinstance(agent, DQNAgent):
        state_dict = torch.load(filename)
        agent.qnetwork_local.load_state_dict(state_dict)
    else:
        raise NotImplementedError
