from pathlib import Path
from typing import Union, Tuple

from deep_rl import agent, agent_io


def create(
    agent_name: str, filename: Union[str, Path, Tuple[Path, Path]], **agent_params
):
    agent_name = f"{agent_name.upper()}Agent"
    if agent_name in agent.DQNAgent.__name__:
        new_agent = agent.DQNAgent(**agent_params)
    elif agent_name in agent.DDPGAgent.__name__:
        new_agent = agent.DDPGAgent(**agent_params)
    else:
        raise NotImplementedError(f"{agent_name} not supported")
    agent_io.load(new_agent, filename)
    return new_agent
