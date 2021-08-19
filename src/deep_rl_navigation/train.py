from collections import deque
from typing import List

import numpy as np

from deep_rl_navigation import agent_io
from deep_rl_navigation.agent import Agent
from deep_rl_navigation.environment import Environment


def train(
    agent: Agent,
    environment: Environment,
    n_episodes=2000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    filename=None,
) -> List[float]:
    """Episodic Reinforcement Learning

    Params
    ======
        agent (Agent): the agent to train
        environment (Environment): the environment to train in
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of steps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        filename (str): Optional path to save agent network
    """
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        state = environment.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = environment.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print(
            f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}",
            end="",
        )
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")
        if np.mean(scores_window) >= 200.0:
            print(
                f"\nEnvironment solved in {i_episode - 100:d} episodes!"
                f"\tAverage Score: {np.mean(scores_window):.2f}"
            )
            if filename:
                agent_io.save(agent, filename)
            break
    return scores
