from pkg_resources import parse_version
from unityagents import UnityEnvironment

from deep_rl import __version__


def test_version():
    assert parse_version(__version__) >= parse_version("1.0.0")


def test_unity_environment():
    env = UnityEnvironment(file_name="/usr/local/sbin/Banana.x86_64")
