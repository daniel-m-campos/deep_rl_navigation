# Deep RL Navigation [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

To easily install the package, clone the repository and use a `virtualenv` to pip install the package in developer mode.

```bash
git clone https://github.com/daniel-m-campos/deep_rl_navigation.git
cd deep_rl_navigation
python -m venv venv # make sure Python 3.6
. venv/bin/activate
pip install -e .
```

### Requirements

See `requirements.txt` and `text-requiremnets.txt`.

### Binary dependencies

The package depends on the Banana Navigation Unity Environment. Download and save to the appropriate binary from:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

And then update the `UNITY_BINARY` path in the `__init__.py` file. The default binary location
is `/usr/local/sbin/Banana.x86_64`.

## Usage

The package provides a [Fire](https://github.com/google/python-fire) CLI for training and playing the agent. To see the
basic commands:

```bash
cd deep_rl_navigation
. venv/bin/activate
python -m deep_rl_navigation --help
```

To see what parameters are available, refer to the `train` and `play` functions in `deep_rl_navigation/__main__.py` as well as the `__init__` method of `DQNAgent` in `deep_rl_navigation/agent.py`

### Train

To train an agent in the Banana Unity environment with default parameters, run:

```bash
cd deep_rl_navigation
. venv/bin/activate
python -m deep_rl_navigation train
```

To train with custom parameters, run for example:

```bash
python -m deep_rl_navigation train \
--n_episodes=100 \
--save_path=None \
--image_path=None \
--learning_rate=5e-3
```

### Play

To play an agent in the Banana Unity environment with default parameters, run:

```bash
cd deep_rl_navigation
. venv/bin/activate
python -m deep_rl_navigation play
```

To play with alternative network, run

```bash
python -m deep_rl_navigation play --load_path="path_to_your/network.pth"
```