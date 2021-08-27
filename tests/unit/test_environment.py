import pytest

from deep_rl import environment


def test_create_raises_error():
    with pytest.raises(NotImplementedError) as test:
        env = environment.create("Doesnt exist")


def test_create():
    env = environment.create("continuous_control")
    assert isinstance(env, environment.ContinuousControl)
