import math
import os
import pathlib
import random
import tempfile

import pytest

from deep_rl import plot


@pytest.fixture
def scores():
    random.seed(2021)
    return [math.sqrt(x) + random.uniform(0, 10) for x in range(100)]


def test_performance_show(scores):
    plot.performance(scores)


def test_performance_save(scores):
    with tempfile.TemporaryDirectory() as tempdir:
        plot_path = pathlib.Path(str(tempdir)) / "test_plot.png"
        plot.performance(scores, save_file=plot_path)
        os.path.isfile(plot_path)
