import math

import fire
import torch

from deep_rl import (
    agent,
    environment,
    play as nav_play,
    train as nav_train,
    agent_io,
    plot,
)


def play(
    max_steps=math.inf,
    load_path="data/checkpoint.pth",
    device_type="cpu",
    **agent_params,
):
    device = torch.device(device_type)
    print(f"Training the Agent with {device.type} device")
    agent.DEVICE = device

    navigation_env = environment.NavigationEnv.from_unity_binary(train_mode=False)
    dqn_agent = agent.DQNAgent(
        state_size=navigation_env.observation_space.shape[0],
        action_size=navigation_env.action_space.n,
        **agent_params,
    )
    agent_io.load(dqn_agent, load_path)
    nav_play.play(dqn_agent, navigation_env, max_steps)


def train(
    n_episodes=1000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    save_path="data/checkpoint.pth",
    image_path="img/performance.png",
    device_type="cpu",
    **agent_params,
):
    device = torch.device(device_type)
    print(f"Training the Agent with {device.type} device")
    agent.DEVICE = device

    # TODO: Parameterize Environment
    navigation_env = environment.NavigationEnv.from_unity_binary()
    # TODO: Parameterize Agent
    dqn_agent = agent.DQNAgent(
        state_size=navigation_env.observation_space.shape[0],
        action_size=navigation_env.action_space.n,
        **agent_params,
    )
    scores = nav_train.train(
        dqn_agent,
        navigation_env,
        n_episodes=n_episodes,
        max_t=max_t,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
    )
    if save_path:
        agent_io.save(dqn_agent, save_path)
    plot.performance(scores, save_file=image_path)


fire.Fire()
