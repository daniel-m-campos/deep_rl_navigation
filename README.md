# Deep RL [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Environments

1. [Navigation](docs/Navigation.md)
2. [Continuous Control](docs/ContinuousControl.md)

## Installation

To easily install the package, clone the repository and use a `virtualenv` to pip install the package in developer mode.

```bash
git clone https://github.com/daniel-m-campos/deep_rl.git
cd deep_rl
python -m venv venv # make sure Python 3.6
. venv/bin/activate
pip install -e .
```

### Requirements

See `requirements.txt` and `text-requiremnets.txt`. These are installed during the `pip install` step.

### Binary dependencies

The package depends on the Udacity's Unity Environments. See [Environments](#Environments) for the binary download links.

Once downloaded, update the `UNITY_BINARY` dict in the `__init__.py` file. The default binary locations are:

```python
UNITY_BINARY = {
    "navigation": "/usr/local/sbin/Banana.x86_64",
    "continuous_control": "/usr/local/sbin/Reacher.x86_64",
}
```

## Usage

The package provides a [Fire](https://github.com/google/python-fire) CLI for training and playing the agent. To see the
basic commands:

```bash
cd deep_rl
. venv/bin/activate
python -m deep_rl --help
```

To see what parameters are available, refer to the `train` and `play` functions in `deep_rl/__main__.py` as well as
the `__init__` method of `Agent` implementations in `deep_rl/agent.py`

### Train

To train an agent in the Navigation/Banana Unity environment with default parameters, run:

```bash
cd deep_rl
. venv/bin/activate
python -m deep_rl navigation train
```

To train with custom parameters, run for example:

```bash
python -m deep_rl navigation train \
  --n_episodes=100 \
  --save_path=None \
  --image_path=None \
  --learning_rate=5e-3
```

### Play

To play an agent in the Banana Unity environment with default parameters, run:

```bash
cd deep_rl
. venv/bin/activate
python -m deep_rl navigation play
```

To play with alternative network, run

```bash
python -m deep_rl navigation play --load_path="path_to_your/network.pth"
```