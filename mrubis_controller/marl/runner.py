from mrubis_controller.marl.mrubis_mock_env import MrubisMockEnv
from multi_agent_controller import MultiAgentController
import logging
import matplotlib.pyplot as plt
import os


class Runner:
    def __init__(self, args, logger, shop_distribution, save_model=False, load_models_data=None):
        self.args = args
        self.logger = logger
        self.shop_distribution = shop_distribution
        self.load_models_data = load_models_data  # if not None saved models are loaded
        self.env = MrubisMockEnv()
        self.mac = MultiAgentController(shop_distribution, self.load_models_data)
        self.t = 0
        self.save_model_interval = 10  # interval of saving models
        self.save_model = save_model

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
        self.reset()
        count_till_fixed = {shop: [] for agent in self.shop_distribution for shop in agent}
        while self.t < episodes:
            terminated = False
            observations = self.env.reset()
            while not terminated:
                actions = self.mac.select_actions(observations)

                reward, observations_, terminated, env_info = self.env.step(actions)
                rewards.append(reward)

                self.mac.learn(observations, actions, reward, observations_, terminated)
                observations = observations_

                if not actions:
                    for shop, count in env_info['stats'].items():
                        count_till_fixed[shop].append(count)
                    # print(env_info)

            if self.t % self.save_model_interval == 0:
                self.mac.save_models(self.t)

            self.t += 1
            print(f"episode {self.t} done")
            if self.t % self.save_model_interval == 0:
                if not os.path.exists('logs'):
                    os.makedirs('logs')
                self.build_reward_plot(rewards, f"Rewards till {self.t}", self.t)
                self.build_count_plot(count_till_fixed, f"Tries till {self.t}", self.t)

    def build_reward_plot(self, data, title, episode):
        plt.title(title, fontsize=12)
        for agent in self.shop_distribution:
            for shop in agent:
                plt.plot([d[0][shop] for d in data], label=shop)
        plt.legend()
        plt.savefig(f"./logs/reward_{episode}")
        plt.clf()  # clear current plot

    def build_count_plot(self, data, title, episode):
        plt.title(title, fontsize=12)
        for agent in self.shop_distribution:
            for shop in agent:
                plt.plot(data[shop], label=shop)
        plt.legend()
        plt.savefig(f"./logs/tries_{episode}")
        plt.clf()  # clear current plot


logging.basicConfig()
logger = logging.getLogger('controller')
logger.setLevel(logging.INFO)
# shop_distribution_example = [{'mRUBiS #1', 'mRUBiS #2'}, {'mRUBiS #3'}]
shop_distribution_example = [{'mRUBiS #1'}, {'mRUBiS #2'}]
# load_model = {"start_time": "2022_01_19_12_37", "episode": 1}
load_model = None
Runner(None, logger, shop_distribution_example, save_model=True, load_models_data=load_model).run(100)
