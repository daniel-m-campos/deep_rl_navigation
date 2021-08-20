from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def performance(scores, figsize=(7, 7), save_file: Union[str, Path] = None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(np.arange(len(scores)), scores)
    ax.set_ylabel("Score")
    ax.set_xlabel("Episode #")
    if save_file:
        fig.savefig(save_file)
    else:
        plt.show()
