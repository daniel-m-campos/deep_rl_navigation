from pathlib import Path
from typing import Union, Tuple

import torch

from deep_rl.agent import Agent, DQNAgent, DDPGAgent


def save(agent: Agent, filename: Union[Path, Tuple[Path, Path]]):
    if isinstance(agent, DQNAgent):
        torch.save(agent.qnetwork_local.state_dict(), filename)
    elif isinstance(agent, DDPGAgent):
        actor_checkpoint, critic_checkpoint = filename
        torch.save(agent.actor_local.state_dict(), actor_checkpoint)
        torch.save(agent.critic_local.state_dict(), critic_checkpoint)
    else:
        raise NotImplementedError


def load(agent: Agent, filename: Union[Path, Tuple[Path, Path]]):
    if isinstance(agent, DQNAgent):
        state_dict = torch.load(filename)
        agent.qnetwork_local.load_state_dict(state_dict)
    elif isinstance(agent, DDPGAgent):
        actor_checkpoint, critic_checkpoint = filename
        agent.actor_local.load_state_dict(torch.load(actor_checkpoint))
        agent.critic_local.load_state_dict(torch.load(critic_checkpoint))
    else:
        raise NotImplementedError
