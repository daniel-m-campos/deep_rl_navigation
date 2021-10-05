import math
from pathlib import Path
from typing import Union, Optional, Tuple

import torch

from deep_rl import (
    agent,
    agent_factory,
    environment,
    play as rl_play,
    train as rl_train,
    agent_io,
    plot,
)

ENV_AGENT_MAP = {
    environment.Navigation: agent.DQNAgent,
    environment.ContinuousControl: agent.DDPGAgent,
    environment.Tennis: agent.MADDPGAgent,
}

ENV_GOAL_MAP = {
    environment.Navigation: 12,
    environment.ContinuousControl: 30,
    environment.Tennis: 0.5,
}


def _filename(path, env):
    name = env.__class__.__name__
    if path != "data":
        return path
    elif isinstance(env, environment.Navigation):
        return Path(path) / f"{name}.pth"
    elif isinstance(env, (environment.ContinuousControl, environment.Tennis)):
        actor_filename = Path(path) / f"{name}Actor.pth"
        critic_filename = Path(path) / f"{name}Critic.pth"
        return actor_filename, critic_filename
    else:
        raise NotImplementedError


def _action_size(env):
    return env.action_space.shape[0] if env.action_space.shape else env.action_space.n


def play(
    env_name: str,
    max_steps: int = math.inf,
    load_path: Union[Tuple[str, Optional[str]], Tuple[Path, Optional[Path]]] = None,
    device_type: str = "cpu",
    **agent_params,
):
    """Play Agent in an Environment

    Params
    ======
        env_name: The environment to train in: "Navigation" or "ContinuousControl"
        max_steps: Maximum number of steps per episode
        load_path: Path to load agent network. If None, default is used
        device_type: Torch device to use
        agent_params: Agent specific parameter overrides
    """

    device = torch.device(device_type)
    print(f"Training the Agent with {device.type} device")
    agent.DEVICE = device

    env = environment.create(env_name, train_mode=False)
    new_agent = agent_factory.create(
        ENV_AGENT_MAP[env.__class__].__name__,
        _filename("data" if load_path is None else load_path, env),
        state_size=env.observation_space.shape[0],
        action_size=_action_size(env),
        **agent_params,
    )
    rl_play.play(new_agent, env, max_steps)


def train(
    env_name: str,
    n_episodes: int = 1000,
    max_steps: int = 1000,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 0.995,
    save_path: Union[Tuple[str, Optional[str]], Tuple[Path, Optional[Path]]] = "data",
    image_path: str = "img",
    device_type: str = "cpu",
    binary_path: str = None,
    **agent_params,
):
    """Train Agent in an Environment

    Params
    ======
        env_name: The environment to train in: "Navigation" or "ContinuousControl"
        n_episodes: Maximum number of training episodes
        max_steps: Maximum number of steps per episode
        eps_start: Starting value of epsilon, for epsilon-greedy action selection
        eps_end: Minimum value of epsilon
        eps_decay: Multiplicative factor (per episode) for decreasing epsilon
        save_path: Path to save agent network. Setting to None disables saving
        image_path: Path to save performance plot
        device_type: Torch device to use
        binary_path: Path Unity environment binary/executable
        agent_params: Agent specific parameter overrides
    """

    device = torch.device(device_type)
    print(f"Training the Agent with {device.type} device")
    agent.DEVICE = device

    env = environment.create(env_name, binary_path)
    new_agent = agent_factory.create(
        ENV_AGENT_MAP[env.__class__].__name__,
        state_size=env.observation_space.shape[0],
        action_size=_action_size(env),
        **agent_params,
    )
    scores = rl_train.train(
        new_agent,
        env,
        n_episodes=n_episodes,
        max_steps=max_steps,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
    )
    if save_path is not None:
        agent_io.save(new_agent, _filename(save_path, env))
    if image_path is not None:
        plot.performance(
            scores,
            save_file=Path(image_path)
            / f"{env.__class__.__name__.lower()}_performance.png",
            goal=ENV_GOAL_MAP[env.__class__],
        )
