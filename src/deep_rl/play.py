import math

from deep_rl.agent import Agent
from deep_rl.environment import Environment
from deep_rl.util import aggregate, is_done


def play(agent: Agent, environment: Environment, max_steps=math.inf) -> float:
    """Play agent in environment

    Params
    ======
        agent (Agent): the agent to play
        environment (Environment): the environment to play in
        max_steps (int): maximum number of steps
    """
    score, steps, done = 0, 0, False
    state = environment.reset()
    while steps < max_steps and not is_done(done):
        steps += 1
        action = agent.act(state)
        state, reward, done, _ = environment.step(action)
        score += aggregate(reward)
    return score
