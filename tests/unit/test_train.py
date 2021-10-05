import deep_rl.util


def test_aggregate_with_float():
    reward = 42.0
    assert reward == deep_rl.util.aggregate(reward, lambda x: -x)


def test_aggregate_with_list():
    rewards = [42.0, 19.0]
    assert deep_rl.util.aggregate(rewards) == rewards[0]


def test_is_done_with_bool():
    done = True
    assert done == deep_rl.util.is_done(done)


def test_is_done_with_list():
    done = [False, False]
    assert not deep_rl.util.is_done(done)
