from mrubis_controller.marl.helper import build_reward_plot, build_count_plot, build_loss_plot, get_current_time
from mrubis_controller.marl.mrubis_mock_env import MrubisMockEnv
from multi_agent_controller import MultiAgentController
import os


class Runner:
    def __init__(self, args, env, shop_distribution, shop_config, save_model=False, load_models_data=None,
                 robustness_activated=False):
        self.args = args
        self.shop_distribution = shop_distribution
        self.load_models_data = load_models_data  # if not None saved models are loaded
        self.env = env
        self.mac = MultiAgentController(self.shop_distribution, self.load_models_data, robustness_activated)
        self.t = 0
        self.inner_t = 0
        self.save_model_interval = 100  # interval of saving models
        self.save_model = save_model
        self.base_dir = f"./mrubis_controller/marl/data/logs/{get_current_time()}"

        self.env.set_shop_config(shop_config)

    def reset(self):
        """ reset all variables and init env """
        self.t = 0
        self.env.reset()
        self.mac.init(self.env.action_space)

    def close_env(self):
        self.env.close()

    def run(self, episodes, test_mode=False, switch=None):
        """ runs the simulation """
        rewards = []
        metrics = []
        self.reset()
        count_till_fixed = {shop: [] for agent in self.shop_distribution for shop in agent}
        while self.t < episodes:
            terminated = False
            if switch is not None and (self.t == episodes / 2 or self.t == episodes / 2 + 1):
                print("NEW CONFIG")
                self.env.set_shop_config(switch)
            observations = self.env.reset()
            while not terminated:
                self.inner_t += 1
                actions = self.mac.select_actions(observations)

                reward, observations_, terminated, env_info = self.env.step(actions)
                print(reward)
                print(observations_)
                print(terminated)
                print(env_info)
                rewards.append(reward)

                metrics.append(self.mac.learn(observations, actions, reward, observations_, terminated))
                observations = observations_

                if self.inner_t > 200:
                    terminated = True
                    print("Forced termination")

                if terminated:
                    self.inner_t = 0
                    for shop, count in env_info['stats'].items():
                        count_till_fixed[shop].append(count)

            self.t += 1
            if self.t % self.save_model_interval == 0:
                if not os.path.exists(self.base_dir):
                    os.makedirs(self.base_dir)
                self.mac.save_models(self.t)
                build_reward_plot(self.base_dir, rewards, self.t, self.shop_distribution)
                build_count_plot(self.base_dir, count_till_fixed, self.t, self.shop_distribution)
                build_loss_plot(self.base_dir, metrics, self.t, self.shop_distribution)
            print(f"episode {self.t} done")


mock_env = MrubisMockEnv(number_of_shops=2)
# shop_distribution_example = [{'mRUBiS #1', 'mRUBiS #2'}, {'mRUBiS #3'}]
# shop_distribution_example = [{'mRUBiS #1'}, {'mRUBiS #2'}, {'mRUBiS #3'}]
shop_distribution_example = [{'mRUBiS #1'}, {'mRUBiS #2'}]
shop_config_example = [[1, 0, False], [2, 2, False]]
load_model = {0: {'start_time': '2022_02_02_13_13', 'episode': 1300}, 1: {'start_time': '2022_02_02_13_13', 'episode': 1300}}
# load_model = {0: None, 1: None, 2: None}
# load_model = {0: None, 1: None}
Runner(None, mock_env, shop_distribution_example, shop_config_example, save_model=True, load_models_data=load_model,
       robustness_activated=True).run(15000, switch=[[2, 2, False], [2, 2, False]])
