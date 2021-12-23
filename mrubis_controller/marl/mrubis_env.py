import gym


class MrubisEnv(gym.Env):
    def __init__(self):
        super(MrubisEnv, self).__init__()
        self.action_space = None
        self.observation_space = None

    def step(self, actions):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    def reset(self):
        """ Returns initial observations and states """
        raise NotImplementedError

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
        raise NotImplementedError
