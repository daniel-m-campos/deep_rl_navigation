import math

import fire
import torch

from deep_rl_navigation import (
    agent,
    environment,
    play as nav_play,
    train as nav_train,
    agent_io,
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
    agent.DEVICE = torch.device("cpu")
    navigation_env = environment.NavigationEnv.from_unity_binary(train_mode=False)
    dqn_agent = agent.DQNAgent(
        state_size=navigation_env.observation_space.shape[0],
        action_size=navigation_env.action_space.n,
        **agent_params,
    )
    agent_io.load(dqn_agent, load_path)
    nav_play.play(dqn_agent, navigation_env, max_steps)


def train(
    n_episodes=2000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    save_path="data/checkpoint.pth",
    device_type="cpu",
    **agent_params,
):
    device = torch.device(device_type)
    print(f"Training the Agent with {device.type} device")
    agent.DEVICE = device

    navigation_env = environment.NavigationEnv.from_unity_binary()
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
        filename=save_path,
    )
    # TODO: plot scores
    print(scores)


fire.Fire()
