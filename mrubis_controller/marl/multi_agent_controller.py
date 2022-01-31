from mrubis_controller.marl.agent import Agent as Agent, encode_observations
from rank_learner import RankLearner
from pathlib import Path
import numpy as np


class MultiAgentController:
    def __init__(self, shop_distribution, load_models_data):
        # list of named shops per agent identified by the index
        self.shop_distribution = shop_distribution
        self.load_models_data = load_models_data
        self.rank_learner = RankLearner(1, None)
        self.agents = None
        self.ridge_regression_train_data_path = Path('../../data/TrainingmRUBiS_Theta0.05_NonStationary.csv')
        self.metrics = []
        self.replay_buffer = []
        self.robustness = {
            'activated': False,  # activates or deactivates the robustness feature
            'actor_lower_threshold': 0,  # value against which the specified statistic is compared
            'actor_upper_threshold': 0.75,  # value against which the specified statistic is compared
            'critic_lower_threshold': 5,  # value against which the specified statistic is compared
            'critic_upper_threshold': 15,  # value against which the specified statistic is compared
            'evaluation_periods': 5,  # number of periods over which data is compared to the specified threshold
            'datapoints_to_alarm': 3,
            # number of data points within the Evaluation Periods that must be breaching to cause the alarm
            'retrain_episodes': 20,
        }
        self.agents_status = {}
        # OK –                  The metric or expression is within the defined threshold.
        # ALARM –               The metric or expression is outside of the defined threshold.
        # INSUFFICIENT_DATA –   The alarm has just started, the metric is not available,
        #                       or not enough data is available for the metric to determine the alarm state.
        # CALIBRATION           The agent is currently trying to recalibrate due to a prior ALARM status
        # RETRAIN               The agent is trying to improve
        # RETIRED               The agent has no shops left and has no tasks to do and is waiting for challenges
        self.calibration_distribution = {}  # stores the shop distribution during calibration mode
        self.retrain_count = {}

    def select_actions(self, observations):
        """ based on observations select actions
            each agent must be called to determine the actions per agent
            the rank learner has to be called to sort those actions
            returns a sorted list of actions
        """
        if self.robustness['activated']:
            # before selecting an action all agents have to be checked if they are in ALARM and init the calibration
            self._update_calibration_distribution()

        # after the calibration is done the agents are asked to perform the action selection
        actions = []
        for index, agent in enumerate(self.agents):
            if self.robustness['activated'] and self.agents_status[index] == 'CALIBRATION':
                # don't use an agent that is currently in calibration mode
                continue
            elif self.robustness['activated'] and self._is_agent_in_charge_for_other_calibrating_agent(index):
                failing_agent_shops = self._get_shops_of_agent_in_charge_for_other_calibrating_agent(index)
                actions.append(
                    agent.choose_action(
                        self._build_observations(index, observations, set.union(*failing_agent_shops['shops']))))
            else:
                actions.append(agent.choose_action(self._build_observations(index, observations)))

        return self.rank_learner.sort_actions(actions)

    def save_models(self, episode):
        """ save models of agents and rank learner """
        for agent in self.agents:
            agent.save(episode)

    def load_models(self):
        """ init models of agents and rank learner
            is loaded when creating the agent and the correct parameters are set
        """
        raise NotImplementedError

    def learn(self, observations, actions, rewards, observations_, dones):
        """ start learning for Agents and RankLearner """
        metrics = {}
        self.replay_buffer.append({
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'observations_': observations_,
            'dones': dones
        })
        for index, agent in enumerate(self.agents):

            # recheck if calibration was correct or wrong
            # if calibration was correct: move shops to other agents by updating shop_distribution
            # and learn action for new agent
            # if calibration was wrong
            # don't learn this and skip
            # update calibration_distribution and test next agent
            # if self.robustness['activated'] and self.agents_status[index] == 'CALIBRATION':
            #     continue
            if self.robustness['activated'] and self._is_agent_in_charge_for_other_calibrating_agent(index):
                # check if other agent was doing a better job
                failing_agent_shops = self._get_shops_of_agent_in_charge_for_other_calibrating_agent(index)
                for n in range(len(failing_agent_shops['failing_agent_index'])):
                    added_shops = failing_agent_shops['shops'][n].difference(agent.shops)
                    for shop in added_shops:
                        if sum(encode_observations(self._build_observations(index, observations_, {shop})[shop])) > 0:
                            # still an error
                            self.calibration_distribution[failing_agent_shops['failing_agent_index'][n]].popitem()

                            # check if all agents were not able to help the agent
                            if len(self.calibration_distribution[
                                       failing_agent_shops['failing_agent_index'][n]].values()) == 0:
                                self.agents_status[failing_agent_shops['failing_agent_index'][n]] = 'RETRAIN'
                        else:
                            self._move_shops(failing_agent_shops['failing_agent_index'][n], index, added_shops)

            metric = agent.learn(self._build_observations(index, observations),
                                 self._build_actions(index, actions),
                                 self._build_rewards(index, rewards),
                                 self._build_observations(index, observations_),
                                 dones)
            metrics[index] = metric
        self.metrics.append(metrics)
        if self.robustness['activated']:
            self._validate_agents()
        return metrics

    def init(self, action_space):
        self._build_agents(action_space)

    def _build_agents(self, action_space):
        """ based on shop distribution the agents will be initialized """
        self.agents = []
        for index, shops in enumerate(self.shop_distribution):
            self.agents.append(
                Agent(index, shops, action_space, self.load_models_data[index], self.ridge_regression_train_data_path))
            self.agents_status[index] = 'INSUFFICIENT_DATA'

    def _build_observations(self, agent_index, observation, shops=None):
        """ extracts the relevant observations of the env per agent """
        relevant_shops = self.agents[agent_index].shops if shops is None else shops
        return {shop: observation[shop] for shop in relevant_shops}

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

    def _validate_agents(self):
        """
        robustness feature
        validate metrics and check if the network fails the criteria
        """
        eval_periods = self.robustness['evaluation_periods']
        for index, agent in enumerate(self.agents):
            if self.agents_status[index] in ['ALARM', 'CALIBRATION', 'RETIRED']:
                continue
            if self.agents_status[index] in ['RETRAIN']:
                # after x steps, the agent should move to OK or ALARM to enable validation again
                self.retrain_count[index] += 1
                if self.retrain_count[index] < self.robustness['retrain_episodes']:
                    continue
                else:
                    # reset retrain count
                    # status will be set to ALARM or OK following the code below
                    self.retrain_count[index] = 0
            agent_metric = [agent_metric for metric in self.metrics for agent_metric in metric[index]][-eval_periods:]
            if len(agent_metric) < eval_periods:
                self.agents_status[index] = 'INSUFFICIENT_DATA'
            else:
                # check if any NN is breaching the defined threshold
                # at least datapoints_to_alarm steps must breach the threshold to set an ALARM
                # otherwise the status is OK
                # TODO: why does the critic loss change by changing the data generator?
                for network in ['actor']:
                    breached_metrics = [item[network] for item in agent_metric if
                                        self.robustness[f"{network}_lower_threshold"] > abs(item[network]) or
                                        abs(item[network]) > self.robustness[f"{network}_upper_threshold"]]
                    if len(breached_metrics) >= self.robustness['datapoints_to_alarm']:
                        self.agents_status[index] = 'ALARM'
                        break
                    else:
                        self.agents_status[index] = 'OK'

    def _init_calibration_distribution(self, failing_agent_index, failing_agent):
        relevant_history = self._build_replay_buffer(failing_agent_index)
        agents_predictions = {}
        for index, agent in enumerate(self.agents):
            if agent != failing_agent:
                # ignore agents similar to failing agent
                if not self._is_agent_check_plausible(agent, relevant_history):
                    continue

                # ignore agents in alarm or calibration state
                if self.agents_status[index] not in ['OK']:
                    continue

                # calc predictions for last step by all agents
                agent_predictions = {}
                for shop_name, components in relevant_history['observations'].items():
                    probabilities = agent.get_probabilities_for_observations(components)
                    max_value = np.amax(probabilities)
                    agent_predictions[shop_name] = max_value
                agents_predictions[index] = agent_predictions

        # rank possible agents bei certainty of action probability
        ranked_agents = self._rank_agents_predictions(relevant_history['observations'], agents_predictions)

        # init calibration_distribution
        ranked_test_distributions = {
            agent_index: {
                'shops': self.agents[agent_index].shops.union(self.agents[failing_agent_index].shops)
            }
            for agent_index in ranked_agents.keys()
        }

        if not ranked_test_distributions:
            # no agent is available to start a calibration so retrain that agent
            self.agents_status[failing_agent_index] = 'RETRAIN'
            self.retrain_count[failing_agent_index] = 0
        else:
            self.calibration_distribution[failing_agent_index] = ranked_test_distributions
            self.agents_status[failing_agent_index] = 'CALIBRATION'

    def _update_calibration_distribution(self):
        """ """
        for index, agent in enumerate(self.agents):
            if self.robustness['activated'] and self.agents_status[index] == 'ALARM':
                #  a) retrain agent
                #  b) ask others agents critic to judge the last step taken -> lowest critic loss
                #  c) or just ask another policy and take the one that would have solved the last evaluation_periods
                #     steps correctly
                #  d) if b) or c) -> switch weights
                #  e) calibration run for testing other agent (rerun on previous data -> replay buffer)
                #  either train_mode: return real state from mrubis as well
                #  or: rerun on wrongly predicted observation:
                #       - does policy give any information? could be that policy is 100% sure, but it's completely wrong
                #  or: order agents by their certainty and ask mRUBiS
                #  f) if no agent is performing great, retrain a new agent (retrain mode)
                #  by copying the old weights and transfer the shops over to this agent
                self._init_calibration_distribution(index, agent)

    def _is_agent_check_plausible(self, agent, relevant_history):
        """ filter agents which would have chosen the exact same action that was wrong """
        actions = agent.choose_action(relevant_history['observations'])
        actions = {action['shop']: action['component'] for action in actions}
        if any(actions[historic_action['shop']] == historic_action['component'] for historic_action in
               relevant_history['actions'].values()):
            return False
        return True

    def _build_replay_buffer(self, index):
        """ extracts only the latest history entry that has resulted in an ALARM """
        for history in self.replay_buffer[-1:]:
            return {
                'observations': self._build_observations(index, history['observations']),
                'actions': self._build_actions(index, history['actions']),
                'rewards': self._build_rewards(index, history['rewards']),
                'observations_': self._build_observations(index, history['observations_']),
                'dones': history['dones']
            }

    def _rank_agents_predictions(self, observations, agents_predictions):
        ranking_per_shop = {}
        # each agent gets points per ranking per shop
        # the less points the better
        score_per_agent = {agent_index: 0 for agent_index, _ in agents_predictions.items()}  # init scores
        for shop_name, components in observations.items():
            # prepare data for sorting
            max_prediction_per_agent = [{'agent_index': agent_index, 'amax_prediction': predictions[shop_name]} for
                                        agent_index, predictions in agents_predictions.items()]
            # sort predictions
            ranking_per_shop[shop_name] = sorted(max_prediction_per_agent, key=lambda d: d['amax_prediction'],
                                                 reverse=True)
        for shop, ranked_predictions in ranking_per_shop.items():
            for index, prediction in enumerate(ranked_predictions):
                score_per_agent[prediction['agent_index']] += index
        return dict(sorted(score_per_agent.items(), key=lambda item: item[1]))

    def _is_agent_in_charge_for_other_calibrating_agent(self, index):
        """
        checks if index_agent is next in line and has currently a failing component so the agent can have a challenge
        """
        return any(len(distribution.keys()) > 0 and index == list(distribution.keys())[0] for distribution in
                   self.calibration_distribution.values())

    def _get_shops_of_agent_in_charge_for_other_calibrating_agent(self, agent_index):
        """
        get all distributions if agent_index is next in line and the failing agent has shops with failures
        so we can have challenge for that agent, otherwise take the next agent
        """
        # TODO: return only shops that have failures
        calibration_distribution = {'shops': [], 'failing_agent_index': []}
        for failing_agent_index, distribution in self.calibration_distribution.items():
            if len(distribution.keys()) > 0 and agent_index == list(distribution.keys())[0]:
                calibration_distribution['shops'].append(list(distribution.values())[0]['shops'])
                calibration_distribution['failing_agent_index'].append(failing_agent_index)
        return calibration_distribution or None

    def _move_shops(self, source_index, destination_index, shops):
        self.agents[source_index].remove_shops(shops)
        self.agents[destination_index].add_shops(shops)

        # check if agent has any shops left
        if len(self.agents[source_index].shops) == 0:
            self.agents_status[source_index] = 'RETIRED'
