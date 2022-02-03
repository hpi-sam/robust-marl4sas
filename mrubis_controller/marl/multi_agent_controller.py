from agent import Agent
from rank_learner import RankLearner


class MultiAgentController:
    def __init__(self, shop_distribution, load_models_data):
        # list of named shops per agent identified by the index
        self.shop_distribution = shop_distribution
        self.load_models_data = load_models_data
        self.rank_learner = RankLearner(0, None)
        self.agents = None

    def select_actions(self, observations):
        """ based on observations select actions
            each agent must be called to determine the actions per agent
            the rank learner has to be called to sort those actions
            returns a sorted list of actions
        """
        actions = []
        for index, agent in enumerate(self.agents):
            actions.append(agent.choose_action(self._build_observations(index, observations)))

        return self.rank_learner.sort_actions(actions)

    def save_models(self, episode):
        """ save models of agents and rank learner """
        for agent in self.agents:
            agent.save(episode)
        self.rank_learner.save(episode)

    def load_models(self):
        """ init models of agents and rank learner
            is loaded when creating the agent and the correct parameters are set
        """
        raise NotImplementedError

    def learn(self, states, actions, rewards, states_, dones):
        """ start learning for Agents and RankLearner """
        for index, agent in enumerate(self.agents):
            # calc threshold
            # if threshold > defined_:
            print(f"Learning for agent {index}.")
            agent.learn(self._build_observations(index, states),
                        self._build_actions(index, actions),
                        self._build_rewards(index, rewards),
                        self._build_observations(index, states_),
                        dones)

    def init(self, action_space):
        self.agents = self._build_agents(action_space)

    def _build_agents(self, action_space):
        """ based on shop distribution the agents will be initialized """
        agents = []
        for index, shops in enumerate(self.shop_distribution):
            agents.append(Agent(index, shops, action_space, self.load_models_data))
        return agents

    def _build_observations(self, agent_index, observation):
        """ extracts the relevant observations of the env per agent """
        return {shop: observation[shop] for shop in self.agents[agent_index].shops}

    def _build_rewards(self, agent_index, rewards):
        """ extracts the relevant rewards per agent
            probably not needed as mRubis is returning reward per shop already
        """
        return {shop: rewards[0][shop] for shop in self.agents[agent_index].shops}, rewards[1]

    def _build_actions(self, agent_index, actions):
        """ extracts the relevant actions taken per agent
        """
        return {index: action
                for shop in self.agents[agent_index].shops
                for index, action in enumerate(actions.values())
                if action['shop'] == shop}
