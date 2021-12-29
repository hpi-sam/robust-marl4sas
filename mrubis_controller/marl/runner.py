from multi_agent_controller import MultiAgentController
from mrubis_env import MrubisEnv


class Runner:
    def __init__(self, args, logger, shop_distribution):
        self.args = args
        self.logger = logger
        self.env = MrubisEnv()
        self.mac = MultiAgentController(shop_distribution)
        self.t = 0

    def reset(self):
        """ reset all variables and init env """
        self.env.reset()
        self.t = 0

    def close_env(self):
        self.env.close()

    def run(self, steps, test_mode=False):
        """ runs the simulation """
        self.reset()
        episode_return = 0
        observations_ = None

        while self.t < steps:
            observations, _, _, _ = self.env.last()

            actions = self.mac.select_actions(observations)

            reward, terminated, env_info = self.env.step(actions)
            episode_return += reward

            self.mac.learn(observations, actions, reward, observations_, terminated)

            # check if this is correct when testing
            observations_ = observations

            self.t += 1
