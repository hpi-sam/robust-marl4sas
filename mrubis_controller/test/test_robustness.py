from mrubis_controller.marl.mrubis_mock_env import MrubisMockEnv
from mrubis_controller.marl.multi_agent_controller import MultiAgentController


def run_one_step(env, mac):
    terminated = False
    observations = env.reset()
    while not terminated:
        actions = mac.select_actions(observations)

        reward, observations_, terminated, env_info = env.step(actions)

        mac.learn(observations, actions, reward, observations_, terminated)
        observations = observations


def test_robustness():
    shop_distribution = [{'mRUBiS #1'}, {'mRUBiS #2'}, {'mRUBiS #3'}]
    load_model = {0: None, 1: None, 2: None}
    # load_model = {0: {'start_time': '2022_01_28_19_03', 'episode': 600}, 1: None, 2: None}
    mac = MultiAgentController(shop_distribution, load_model, True)
    mock_env = MrubisMockEnv(number_of_shops=3, shop_config=[1, 0, False])

    # init test setup
    mock_env.reset()
    mac.init(mock_env.action_space)

    assert all(
        status == 'INSUFFICIENT_DATA'
        for index, status in mac.robustness.agents_status.items()
    )

    # after 1 step
    run_one_step(mock_env, mac)
    assert mac.robustness.agents_status[0] == 'OK'

    # after 10 steps
    for _ in range(9):
        run_one_step(mock_env, mac)
    assert mac.robustness.agents_status[0] == 'OK'
    assert any(
        status == 'RETIRED'
        for index, status in mac.robustness.agents_status.items()
    )
