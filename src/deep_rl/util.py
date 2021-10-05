from typing import Union, List, Callable


def aggregate(reward: Union[float, List[float]], score_agg: Callable = max):
    if isinstance(reward, List):
        return score_agg(reward)
    return reward


def is_done(done: Union[bool, List[bool]]):
    if isinstance(done, List):
        return any(done)
    return done
