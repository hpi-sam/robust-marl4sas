import gym
import json
import os


def get_failing_component(current_shop):
    """ returns the current failing component or None if no failure available """
    for component in current_shop:
        if component['failure_name'] != "None":
            return component
    return None


def get_observation(step):
    """ returns the observation for the given step """
    with open(os.path.dirname(__file__) + f"/data/test/observation_step_{step}.json") as json_data_file:
        return json.load(json_data_file)


def get_reward(observation):
    """ returns the extracted reward per shop """
    return [{'shop1': 1000}]


class MrubisEnv(gym.Env):
    def __init__(self):
        super(MrubisEnv, self).__init__()
        self.action_space = None
        self.observation_space = None
        self.observation = None
        self.step = 1

    def step(self, actions):
        """ Returns reward, terminated, info
            see /data-analysis/step_example.json for an example

            increases component_utility if fix is correct
            decreases component_utility if fix is wrong
        """
        for action in actions:
            current_shop = self.observation[action['shop']]
            current_failing_component = get_failing_component(current_shop)
            if current_failing_component is not None:
                if current_failing_component == action:
                    current_failing_component['failure_name'] = 'None'
                    current_failing_component['component_utility'] = 10000
                    # shop utility must be increased as well
                else:
                    current_failing_component['component_utility'] -= 10000
                    # shop utility must be decreased as well
            else:
                # action for a shop without any failure
                raise NotImplementedError
        return self.observation, 0, None

    def reset(self):
        """ Returns initial observations and states """
        self.step = 1
        self.observation = get_observation(self.step)
        self.action_space = [value.keys() for key, value in self.observation['observations'][0].items()]
        return self.observation

    def render(self):
        """ Renders the environment. """
        raise NotImplementedError

    def close(self):
        """ Override close in your subclass to perform any necessary cleanup. """
        raise NotImplementedError

    def seed(self):
        """ Sets the seed for this env's random number generator(s). """
        raise NotImplementedError

    def last(self):
        """ returns last state, reward, terminated, info """
        return self.observation, get_reward(self.observation), 0, None


mrubis_env = MrubisEnv()
mrubis_env.reset()
print(mrubis_env.last())
