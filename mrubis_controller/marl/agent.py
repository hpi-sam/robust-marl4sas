class Agent:
    def __init__(self):
        self.actor = None
        self.critic = None
        self.policy = None
        self.args = None

    def choose_action(self, observations):
        """ chooses actions based on observations
            each trace will be introduced to the network to find a fix
            sort actions afterwards for maximizing utility
            returns sorted list
        """
        raise NotImplementedError

    def learn(self, states, actions, rewards, states_, dones):
        """ network learns to improve """
        raise NotImplementedError

    def _sort_actions(self, actions):
        """ sort actions with RidgeRegression
            returns sorted list
        """
        raise NotImplementedError

    def _build_network(self):
        raise NotImplementedError
