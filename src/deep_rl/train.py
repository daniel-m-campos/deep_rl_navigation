from collections import deque
from typing import List

import numpy as np

from deep_rl.agent import Agent
from deep_rl.environment import Environment


def train(
    agent: Agent,
    environment: Environment,
    n_episodes=2000,
    max_steps=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    max_score=True,
) -> List[float]:
    """Episodic Reinforcement Learning

    Params
    ======
        agent (Agent): the agent to train
        environment (Environment): the environment to train in
        n_episodes (int): maximum number of training episodes
        max_steps (int): maximum number of steps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        filename (str): Optional path to save agent network
        max_score (bool): Optional use max score if reward is vector
    """
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        state = environment.reset()
        score = 0
        for t in range(max_steps):
            action = agent.act(state, eps)
            next_state, reward, done, _ = environment.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward if len(reward) == 1 else reward.max()
            if any(done):
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
    return scores
