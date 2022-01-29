import gym
# import json
# import os
from mrubis_controller.marl.data.data_generator import DataGenerator
import copy


def get_failing_component(current_shop):
    """ returns the current failing component or None if no failure available """
    for component in current_shop:
        if current_shop[component]['failure_name'] != "None":
            return component
    return None


def get_current_utility(observation):
    return {shop: float(list(components.items())[0][1]['shop_utility']) for shop, components in
            observation.items()}


class MrubisMockEnv(gym.Env):
    def __init__(self, number_of_shops=3, shop_config=None):
        super(MrubisMockEnv, self).__init__()
        if shop_config is None:
            shop_config = [1, 0, False]
        self.number_of_shops = number_of_shops
        self.action_space = None
        self.observation_space = None
        self.failing_components = None
        self.observation = None
        self.prior_utility = None  # used for calculation of diff as a reward
        self.t = 0
        self.termination_t = 3
        self.inner_t = 0
        self.stats = {}
        self.terminated = False

        self.utility_decrease_amount = 1  # if fix fails
        self.utility_increase_amount = 17  # if fix succeeds
        self.data_generator = DataGenerator(number_of_shops=number_of_shops)
        self.data_generator.set_shop_config(shop_config)

    def _get_observation(self, step):
        """ returns the observation and actual state for the given step """
        return self.data_generator.generate_shops_with_failures(step)

    def step(self, actions):
        """ Returns observation, reward, terminated, info
            see /data-analysis/step_example.json for an example

            increases component_utility if fix is correct
            decreases component_utility if fix is wrong
        """
        self.inner_t += 1
        for action in actions.values():
            current_shop = self.observation[action['shop']]
            current_failing_component = self.failing_components[action['shop']]
            if current_failing_component is None:
                # action for a shop without any failure
                raise NotImplementedError
            if current_failing_component == action['component']:
                for component in current_shop:
                    current_shop[component]['failure_name'] = 'None'
                current_shop[current_failing_component]['component_utility'] = float(
                    current_shop[current_failing_component]['component_utility']) + self.utility_increase_amount
                # first component holds information of utility
                # shop utility must be increased as well
                current_shop['Availability Item Filter']['shop_utility'] = float(
                    current_shop['Availability Item Filter']['shop_utility']) + self.utility_increase_amount
                self.stats[action['shop']] = self.inner_t
            else:
                current_shop[current_failing_component]['component_utility'] = float(
                    current_shop[current_failing_component]['component_utility']) - self.utility_decrease_amount
                # first component holds information of utility
                # shop utility must be decreased as well
                current_shop['Availability Item Filter']['shop_utility'] = float(
                    current_shop['Availability Item Filter']['shop_utility']) - self.utility_decrease_amount
        _reward = self._get_reward(self.observation)
        # check if all issues are fixed and load new observation if all are fixed
        if self.all_fixed(actions):
            self.t += 1
            self.inner_t = 0
            self.terminated = True
            # TODO: check with Ulrike whether we need this if check
            self.observation, self.failing_components = self._get_observation(self.t)
            self.prior_utility = get_current_utility(self.observation)
        return _reward, copy.deepcopy(self.observation), self.terminated, self._info()

    def all_fixed(self, actions):
        return not actions or all(failure is None for failure in
                                  [get_failing_component(self.observation[shop]) for shop in self.observation])

    def reset(self):
        """ Returns initial observations and states """
        self.t = 0
        self.terminated = False
        self.observation, self.failing_components = self._get_observation(self.t)
        self.prior_utility = get_current_utility(self.observation)
        self.action_space = [components for shops, components in self.observation.items()][0].keys()
        return copy.deepcopy(self.observation)

    def render(self):
        """ Renders the environment. """
        raise NotImplementedError

    def close(self):
        """ Override close in your subclass to perform any necessary cleanup. """
        return 'do nothing for this mock'

    def seed(self):
        """ Sets the seed for this env's random number generator(s). """
        return 'do nothing for this mock'

    def last(self):
        """ returns last state, reward, terminated, info """
        return self.observation, self._get_reward(self.observation), self.terminated, self._info()

    def _info(self):
        return {'t': self.t, 'stats': self.stats}

    def _get_reward(self, observation):
        """ returns the extracted reward per shop
        """
        current_utility = get_current_utility(observation)
        diff_utility = {shop: current_utility[shop] - utility for shop, utility in self.prior_utility.items()}
        self.prior_utility = current_utility
        system_utility = sum(diff_utility.values())
        return diff_utility, system_utility

# code for local testing/debugging - until test framework is available
# mrubis_env = MrubisEnv()
# mrubis_env.reset()
# # print(mrubis_env.last())
# print(mrubis_env.step({
#     0: {
#         'shop': 'mRUBiS #1',
#         'component': 'Availability Item Filter',
#     },
#     1: {
#         'shop': 'mRUBiS #2',
#         'component': 'Bid and Buy Service',
#     },
#     2: {
#         'shop': 'mRUBiS #3',
#         'component': 'Future Sales Item Filter',
#     }}))
