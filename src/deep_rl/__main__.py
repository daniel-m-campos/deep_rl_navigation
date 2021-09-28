import fire

from deep_rl import main

fire.Fire({"play": main.play, "train": main.train})
