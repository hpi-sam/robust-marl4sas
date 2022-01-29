from mrubis_controller.marl.agent import encode_observations
from mrubis_controller.marl.mrubis_mock_env import MrubisMockEnv
import numpy as np


def test_reset():
    env = MrubisMockEnv(number_of_shops=3)
    observation = env.reset()
    assert len(observation.keys()) == 3
    count_of_issues = len([item for values in observation.values() for component, item in values.items() if
                           item['failure_name'] != 'None'])
    assert count_of_issues == 3


def test_correct_action_for_5_episodes():
    mock_env = MrubisMockEnv(number_of_shops=1, shop_config=[1, 0, False])
    shop = 'mRUBiS #1'
    for _ in range(5):
        observation = mock_env.reset()
        one_hot_state = encode_observations(observation[shop])[np.newaxis, :][0]
        failing_component_from_action_space = list(mock_env.action_space)[np.where(one_hot_state == 1)[0][0]]
        action = {0: {'shop': shop, 'component': failing_component_from_action_space}}
        reward, observations_, terminated, env_info = mock_env.step(action)
        assert reward[0][shop] == 10
        assert terminated is True
        steps_till_solved = 1
        assert env_info['stats'][shop] == steps_till_solved


def test_wrong_action_for_5_episodes():
    mock_env = MrubisMockEnv(number_of_shops=1, shop_config=[1, 0, False])
    shop = 'mRUBiS #1'
    observation = mock_env.reset()
    for _ in range(5):
        one_hot_state = encode_observations(observation[shop])[np.newaxis, :][0]
        wrong_failing_component_from_action_space = list(mock_env.action_space)[np.where(one_hot_state == 0)[0][0]]
        action = {0: {'shop': shop, 'component': wrong_failing_component_from_action_space}}
        reward, observations_, terminated, env_info = mock_env.step(action)
        assert reward[0][shop] == -1
        assert terminated is False
        assert len(env_info['stats'].keys()) == 0
        assert sum(encode_observations(observations_[shop])[np.newaxis, :][0]) == 1
        observation = observations_
