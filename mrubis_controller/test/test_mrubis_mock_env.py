from mrubis_controller.marl.mrubis_mock_env import MrubisMockEnv


def test_reset():
    env = MrubisMockEnv(number_of_shops=3)
    observation = env.reset()
    assert len(observation.keys()) == 3
    count_of_issues = len([item for values in observation.values() for component, item in values.items() if
                           item['failure_name'] != 'None'])
    assert count_of_issues == 3
