from marl.helper import build_reward_plot, build_count_plot, build_loss_plot, get_current_time
from mrubis_controller.marl.mrubis_mock_env import MrubisMockEnv
from multi_agent_controller import MultiAgentController
import os


class Runner:
    def __init__(self, args, env, shop_distribution, save_model=False, load_models_data=None):
        self.args = args
        self.shop_distribution = shop_distribution
        self.load_models_data = load_models_data  # if not None saved models are loaded
        self.env = env
        self.mac = MultiAgentController(self.shop_distribution, self.load_models_data)
        self.t = 0
        self.save_model_interval = 10  # interval of saving models
        self.save_model = save_model
        self.base_dir = f"./data/logs/{get_current_time()}"

    def reset(self):
        """ reset all variables and init env """
        self.t = 0
        self.env.reset()
        self.mac.init(self.env.action_space)

    def close_env(self):
        self.env.close()

    def run(self, episodes, test_mode=False):
        """ runs the simulation """
        rewards = []
        metrics = []
        self.reset()
        count_till_fixed = {shop: [] for agent in self.shop_distribution for shop in agent}
        while self.t < episodes:
            terminated = False
            observations = self.env.reset()
            while not terminated:
                actions = self.mac.select_actions(observations)

                reward, observations_, terminated, env_info = self.env.step(actions)
                rewards.append(reward)

                metrics.append(self.mac.learn(observations, actions, reward, observations_, terminated))
                observations = observations_

                if not actions:
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


mock_env = MrubisMockEnv()
# shop_distribution_example = [{'mRUBiS #1', 'mRUBiS #2'}, {'mRUBiS #3'}]
shop_distribution_example = [{'mRUBiS #1'}, {'mRUBiS #2'}]
load_model = {"start_time": "2022_01_24_08_58", "episode": 40}
# load_model = None
Runner(None, mock_env, shop_distribution_example, save_model=True, load_models_data=load_model).run(100)
