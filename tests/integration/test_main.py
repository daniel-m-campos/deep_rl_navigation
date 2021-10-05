from pathlib import Path

import pytest

from deep_rl import main


@pytest.fixture
def root():
    return Path(__file__).parent.parent.parent


def test_play_navigation(root):
    main.play(
        "navigation",
        max_steps=10,
        load_path=root / "data/Navigation.pth",
    )


def test_play_tennis(root):
    paths = (
        root / "data/TennisActor.pth",
        root / "data/TennisCritic.pth",
    )
    main.play(
        "tennis",
        max_steps=10,
        load_path=paths,
    )


def test_train(root):
    paths = (
        root / "data/TennisActor.pth",
        root / "data/TennisCritic.pth",
    )
    main.train("tennis", save_path=paths, image_path=root / "img")
