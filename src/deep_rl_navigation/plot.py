from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def performance(scores, figsize=(7, 7), save_file: Union[str, Path] = None, goal=13):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    scores = np.array(scores)
    episodes = np.arange(scores.shape[0])
    ax.plot(episodes, scores)
    ax.set_ylabel("Score")
    ax.set_xlabel("Episode #")

    solved_episode = np.argmax(scores > goal)
    ax.hlines(
        goal,
        episodes[1],
        episodes[-2],
        colors="red",
        label=f"Goal {goal} achieved at episode #{solved_episode}",
    )
    ax.legend()
    if save_file:
        fig.savefig(save_file)
    else:
        plt.show()
