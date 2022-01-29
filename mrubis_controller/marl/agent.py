# follows https://dev.to/jemaloqiu/reinforcement-learning-with-tf2-and-gym-actor-critic-3go5

from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from mrubis_controller.marl.sorting.agent_action_sorter import AgentActionSorter
from mrubis_controller.marl.helper import get_current_time


def _decoded_action(action, observation):
    return list([components.keys() for shop, components in observation.items()][0])[action]


def encode_observations(observations):
    encoded_observations = [
        1 if component['failure_name'] != 'None' else 0
        for component in observations.values()
    ]

    return np.array(encoded_observations, dtype=float)


class Agent:
    def __init__(self, index, shops, action_space_inverted, load_models_data, ridge_regression_train_data_path):
        self.index = index
        self.shops = shops
        self.base_model_dir = './data/models'
        self.start_time = get_current_time()
        self.load_models_data = load_models_data

        self.action_space_inverted = list(action_space_inverted)
        self.gamma = 0.99
        self.alpha = 0.0001
        self.beta = 0.0005
        self.n_actions = len(action_space_inverted)
        self.input_dims = self.n_actions
        self.fc1_dims = 36
        self.fc2_dims = 72
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)

        self.actor, self.critic, self.policy = self._build_network()
        self.action_space = list(range(self.n_actions))

        # stage 0 = no sorting as a baseline
        # stage 1 = sorting of actions
        self.stage = 1

        self.agent_action_sorter = AgentActionSorter(ridge_regression_train_data_path)

    def choose_action(self, observations):
        """ chooses actions based on observations
            each trace will be introduced to the network to find a fix
            sort actions afterwards for maximizing utility
            returns sorted list
        """
        actions = []
        for shop_name, components in observations.items():
            state = encode_observations(components)[np.newaxis, :]
            if state.sum() > 0:  # skip shops without any failure
                probabilities = self.policy.predict(state)[0]
                action = np.random.choice(self.action_space, p=probabilities)
                decoded_action = _decoded_action(action, observations)
                step = {'shop': shop_name, 'component': decoded_action}
                if self.stage >= 1:
                    step['predicted_utility'] = self.agent_action_sorter.predict_optimal_utility_of_fixed_components(
                        step, components)
                if self.stage == 2:
                    # reduce predicted utility by uncertainty
                    step['predicted_utility'] *= probabilities[action]
                actions.append(step)
        return actions

    def learn(self, states, actions, reward, states_, dones):
        """ network learns to improve """
        actions = {action['shop']: action['component'] for action in actions.values()}
        metrics = []
        for shop_name, action in actions.items():
            state = encode_observations(states[shop_name])[np.newaxis, :]
            state_ = encode_observations(states_[shop_name])[np.newaxis, :]

            critic_value = self.critic.predict(state)
            critic_value_ = self.critic.predict(state_)

            # TODO: How important is the length of an episode? Is there a future reward of a solved state?
            shop_reward = reward[0][shop_name]
            target = shop_reward + self.gamma * critic_value_ * (1 - int(dones))
            delta = target - critic_value

            _actions = np.zeros([1, self.n_actions])
            _actions[np.arange(1), self.action_space_inverted.index(action)] = 1.0

            critic_history = self.critic.fit(state, target, verbose=0)

            with tf.GradientTape() as tape:
                y_pred = self.actor(state)
                out = K.clip(y_pred, 1e-8, 1 - 1e-8)
                log_lik = _actions * K.log(out)
                actor_loss = K.sum(-log_lik * delta)
            grads = tape.gradient(actor_loss, self.actor.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

            metrics.append({"actor": float(actor_loss), "critic": critic_history.history['loss'][0]})
        return metrics

    def get_probabilities_for_observations(self, observations):
        return self.policy.predict(encode_observations(observations)[np.newaxis, :])[0]

    def _build_network(self):
        if self.load_models_data is not None:
            return self.load_models(self.load_models_data)
        input = Input(shape=(self.input_dims,), name='input')
        delta = Input(shape=[1], name='delta')
        dense1 = Dense(self.fc1_dims, activation='relu', name='dense1')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)

        probs = Dense(self.n_actions, activation='softmax', name='probs')(dense2)
        values = Dense(1, activation='linear', name='values')(dense2)

        actor = Model(inputs=[input, delta], outputs=[probs])

        critic = Model(inputs=[input], outputs=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        policy = Model(inputs=[input], outputs=[probs])

        return actor, critic, policy

    def save(self, episode):
        self.actor.save(f"{self.base_model_dir}/{self.start_time}/agent_{self.index}/actor/episode_{episode}")
        self.critic.save(f"{self.base_model_dir}/{self.start_time}/agent_{self.index}/critic/episode_{episode}")

    def load_models(self, load_models_data):
        base_dir = f"{self.base_model_dir}/{load_models_data['start_time']}/agent_{self.index}"

        # load critic
        critic = tf.keras.models.load_model(f"{base_dir}/critic/episode_{load_models_data['episode']}")

        # load actor and set layers
        actor_copy = tf.keras.models.load_model(f"{base_dir}/actor/episode_{load_models_data['episode']}")
        probs = actor_copy.get_layer('probs')(critic.get_layer('dense1').output)
        actor = Model(inputs=[critic.get_layer('input').input, actor_copy.get_layer('delta').input], outputs=[probs])

        # load policy
        policy = Model(inputs=[critic.get_layer('input').input], outputs=[probs])

        return actor, critic, policy

    def remove_shops(self, shops):
        self.shops = self.shops.difference(shops)

    def add_shops(self, shops):
        self.shops = self.shops.union(shops)
