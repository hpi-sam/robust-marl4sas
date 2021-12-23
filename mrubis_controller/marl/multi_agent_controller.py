from rank_learner import RankLearner


class MultiAgentController:
    def __init__(self, shop_distribution):
        # list of named shops per agent identified by the index
        self.shop_distribution = shop_distribution
        self.rank_learner = RankLearner(0, None)
        self.agents = None

    def select_actions(self, observations):
        """ based on observations select actions
            each agent must be called to determine the actions per agent
            the rank learner has to be called to sort those actions
            returns a sorted list of actions
        """
        raise NotImplementedError

    def save_models(self):
        """ save models of agents and rank learner """
        raise NotImplementedError

    def load_models(self):
        """ init models of agents and rank learner """
        raise NotImplementedError

    def learn(self, states, actions, rewards, states_, dones):
        """ start learning for Agents and RankLearner """
        raise NotImplementedError

    def _build_agents(self):
        """ based on shop distribution the agents will be initialized """
        raise NotImplementedError

    def _build_observations(self, agent_id):
        """ extracts the relevant observations of the env per agent """
        raise NotImplementedError

    def _build_rewards(self, agent_id):
        """ extracts the relevant rewards per agent
            probably not needed as mRubis is returning reward per shop already
        """
        raise NotImplementedError
