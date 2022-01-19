import gym
import json
import os


def get_failing_component(current_shop):
    """ returns the current failing component or None if no failure available """
    for component in current_shop:
        if current_shop[component]['failure_name'] != "None":
            return component
    return None


def get_observation(step):
    """ returns the observation for the given step """
    with open(os.path.dirname(__file__) + f"/data/test/observation_step_{step}.json") as json_data_file:
        return json.load(json_data_file)


class MrubisMockEnv(gym.Env):
    def __init__(self):
        super(MrubisMockEnv, self).__init__()
        self.action_space = None
        self.observation_space = None
        self.observation = None
        self.prior_utility = None  # used for calculation of diff as a reward
        self.t = 0
        self.termination_t = 3

        self.utility_decrease_amount = 1  # if fix fails
        self.utility_increase_amount = 1  # if fix succeeds

    def step(self, actions):
        """ Returns observation, reward, terminated, info
            see /data-analysis/step_example.json for an example

            increases component_utility if fix is correct
            decreases component_utility if fix is wrong
        """
        for action in actions.values():
            current_shop = self.observation[action['shop']]
            current_failing_component = get_failing_component(current_shop)
            if current_failing_component is not None:
                if current_failing_component == action['component']:
                    current_shop[current_failing_component]['failure_name'] = 'None'
                    current_shop[current_failing_component]['component_utility'] = float(
                        current_shop[current_failing_component]['component_utility']) + self.utility_increase_amount
                    # first component holds information of utility
                    # shop utility must be increased as well
                    current_shop['Availability Item Filter']['shop_utility'] = float(
                        current_shop['Availability Item Filter']['shop_utility']) + self.utility_increase_amount
                else:
                    current_shop[current_failing_component]['component_utility'] = float(
                        current_shop[current_failing_component]['component_utility']) - self.utility_decrease_amount
                    # first component holds information of utility
                    # shop utility must be decreased as well
                    current_shop['Availability Item Filter']['shop_utility'] = float(
                        current_shop['Availability Item Filter']['shop_utility']) - self.utility_decrease_amount
            else:
                # action for a shop without any failure
                raise NotImplementedError
        _reward = self._get_reward(self.observation)
        # check if all issues are fixed and load new observation if all are fixed
        if all(failure is None for failure in
               [get_failing_component(self.observation[shop]) for shop in self.observation]):
            self.t += 1
            self.prior_utility = None
            if not self._terminated():
                self.observation = get_observation(self.t)
        return _reward, self.observation, self._terminated(), self._info()

    def reset(self):
        """ Returns initial observations and states """
        self.t = 0
        self.observation = get_observation(self.t)
        self.action_space = [components for shops, components in self.observation.items()][0].keys()
        return self.observation

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
        return self.observation, self._get_reward(self.observation), self._terminated(), self._info()

    def _terminated(self):
        return self.t == self.termination_t

    def _info(self):
        return {'t': self.t}

    def _get_reward(self, observation):
        """ returns the extracted reward per shop
        """
        current_utility = {shop: float(list(components.items())[0][1]['shop_utility']) for shop, components in
                           observation.items()}
        if self.prior_utility is None:
            diff_utility = {shop: 0 for shop, utility in current_utility.items()}
        else:
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
