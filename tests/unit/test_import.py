def test_version():
    from deep_rl_navigation import __version__
    from pkg_resources import parse_version

    assert parse_version(__version__) >= parse_version("0.0.0")


def test_unity_environment():
    from deep_rl_navigation import UNITY_BINARY
    from unityagents import UnityEnvironment

    env = UnityEnvironment(file_name=UNITY_BINARY)
