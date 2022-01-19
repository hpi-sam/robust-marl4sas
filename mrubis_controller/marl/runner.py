from mrubis_controller.marl.mrubis_mock_env import MrubisMockEnv
from multi_agent_controller import MultiAgentController
import logging
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, args, logger, shop_distribution):
        self.args = args
        self.logger = logger
        self.env = MrubisMockEnv()
        self.mac = MultiAgentController(shop_distribution)
        self.t = 0

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
        while self.t < episodes:
            terminated = False
            observations = self.env.reset()
            while not terminated:
                actions = self.mac.select_actions(observations)

                reward, observations_, terminated, env_info = self.env.step(actions)
                rewards.append(reward)

                self.mac.learn(observations, actions, reward, observations_, terminated)
                observations = observations_

            self.t += 1
            self.build_plot(rewards, f"Rewards till {self.t}", self.t)

    def build_plot(self, data, title, episode):
        plt.title(title, fontsize=12)
        for agent in self.shop_distribution:
            for shop in agent:
                plt.plot([d[0][shop] for d in data], label=shop)
        plt.legend()
        plt.savefig(f"./logs/rewards_{episode}")
        plt.clf()  # clear current plot


logging.basicConfig()
logger = logging.getLogger('controller')
logger.setLevel(logging.INFO)
shop_distribution = [{'mRUBiS #1', 'mRUBiS #2'}, {'mRUBiS #3'}]
Runner(None, logger, shop_distribution).run(20)
