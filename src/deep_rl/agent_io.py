from pathlib import Path
from typing import Union, Tuple

import torch

from deep_rl.agent import Agent, DQNAgent, DDPGAgent, MADDPGAgent


def _add_index(checkpoint, i):
    return Path(str(checkpoint).replace(".pth", f"{i}.pth"))


def save(agent: Agent, filename: Union[str, Path, Tuple[Path, Path]]):
    if isinstance(agent, DQNAgent):
        torch.save(agent.qnetwork_local.state_dict(), filename)
    elif isinstance(agent, DDPGAgent):
        actor_checkpoint, critic_checkpoint = filename
        torch.save(agent.actor_local.state_dict(), actor_checkpoint)
        torch.save(agent.critic_local.state_dict(), critic_checkpoint)
    elif isinstance(agent, MADDPGAgent):
        actor_checkpoint, critic_checkpoint = filename
        for i, ddpg_agent in enumerate(agent.agents, start=1):
            torch.save(
                ddpg_agent.actor_local.state_dict(),
                _add_index(actor_checkpoint, i),
            )
            torch.save(
                ddpg_agent.critic_local.state_dict(),
                _add_index(critic_checkpoint, i),
            )
    else:
        raise NotImplementedError


def load(agent: Agent, filename: Union[str, Path, Tuple[Path, Path]]):
    if isinstance(agent, DQNAgent):
        state_dict = torch.load(filename)
        agent.qnetwork_local.load_state_dict(state_dict)
    elif isinstance(agent, DDPGAgent):
        actor_checkpoint, critic_checkpoint = filename
        agent.actor_local.load_state_dict(torch.load(actor_checkpoint))
        agent.critic_local.load_state_dict(torch.load(critic_checkpoint))
    elif isinstance(agent, MADDPGAgent):
        actor_checkpoint, critic_checkpoint = filename
        for i, ddpg_agent in enumerate(agent.agents, start=1):
            ddpg_agent.actor_local.load_state_dict(
                torch.load(_add_index(actor_checkpoint, i))
            )
            ddpg_agent.critic_local.load_state_dict(
                torch.load(_add_index(critic_checkpoint, i))
            )
    else:
        raise NotImplementedError
