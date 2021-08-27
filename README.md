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

The package depends on the Udacity's Unity Environments. See [Environments](#Environments) for the binary download
links.

The default binary paths are set in the `Environment` implementations and are of the
form `/usr/local/sbin/<ENVIRONMENT>.x86_64`. See the `Navigation` class in `environment.py` for an example. You can
either sim link the downloaded binaries to the default directories or pass the `binary_path` when running the package.

## Usage

The package provides a [Fire](https://github.com/google/python-fire) CLI for training and playing the agent. To see the
basic commands:

```bash
cd deep_rl
. venv/bin/activate
python -m deep_rl <command> --help
```

Where `<command>` is either `train` or `play`. See `deep_rl/__main__.py` as well as the `__init__` method of `Agent`
implementations in `deep_rl/agent.py`

### Train

To train an agent in the Navigation/Banana Unity environment with default parameters, run:

```bash
cd deep_rl
. venv/bin/activate
python -m deep_rl train navigation
```

To train with custom parameters, run for example:

```bash
python -m deep_rl train navigation \
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
python -m deep_rl play navigation
```

To play with alternative network, run

```bash
python -m deep_rl play navigation --load_path="path_to_your/network.pth"
```