from deep_rl_navigation import environment, UNITY_BINARY


def test_unity_navigation_constructor():
    unity_env = environment.UnityEnvironment(file_name=UNITY_BINARY)
    env = environment.NavigationEnv(unity_env)
