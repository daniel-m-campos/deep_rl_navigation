from pathlib import Path

from deep_rl import main


def test_play():
    main.play(
        "navigation",
        max_steps=10,
        load_path=Path(__file__).parent.parent.parent / "data/Navigation.pth",
    )


def test_train():
    main.train("tennis")
