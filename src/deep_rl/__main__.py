import math

import fire
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
}


def play(
    env_name: str,
    max_steps: int = math.inf,
    load_path: str = None,
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
    dqn_agent = agent_factory.create(
        ENV_AGENT_MAP[env.__class__].__name__,
        f"data/{env.__class__.__name__}.pth" if load_path is None else load_path,
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        **agent_params,
    )
    rl_play.play(dqn_agent, env, max_steps)


def train(
    env_name: str,
    n_episodes: int = 1000,
    max_steps: int = 1000,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 0.995,
    save_path: str = "",
    image_path: str = "img/performance.png",
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
    dqn_agent = agent_factory.create(
        ENV_AGENT_MAP[env.__class__].__name__,
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        **agent_params,
    )
    scores = rl_train.train(
        dqn_agent,
        env,
        n_episodes=n_episodes,
        max_steps=max_steps,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
    )
    if save_path is not None:
        if save_path == "":
            save_path = f"data/{env.__class__.__name__}.pth"
        agent_io.save(dqn_agent, save_path)
    plot.performance(scores, save_file=image_path)


fire.Fire()
