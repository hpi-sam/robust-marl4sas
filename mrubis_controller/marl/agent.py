from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np

from mrubis_controller.marl.helper import get_current_time

tf.config.experimental_run_functions_eagerly(True)


def _decoded_action(action, observation):
    return list([components.keys() for shop, components in observation.items()][0])[action]


def _encode_observations(observations):
    encoded_observations = []
    for component in observations.values():
        encoded_observations += [1 if component['failure_name'] != 'None' else 0]
    return np.array(encoded_observations, dtype=float)


def _delta_custom_loss(delta):
    def custom_loss(y_true, y_pred):
        out = K.clip(y_pred, 1e-8, 1 - 1e-8)
        log_like = y_true * K.log(out)

        return K.sum(-log_like * delta)

    return custom_loss


class Agent:
    def __init__(self, index, shops, action_space_inverted, load_models_data):
        self.index = index
        self.shops = shops
        self.base_model_dir = f"./data/models"
        self.start_time = get_current_time()
        self.load_models_data = load_models_data

        self.action_space_inverted = list(action_space_inverted)
        self.gamma = 0.99
        self.alpha = 0.00001
        self.beta = 0.00005
        self.n_actions = len(action_space_inverted)
        self.input_dims = self.n_actions
        self.fc1_dims = 1024
        self.fc2_dims = 512

        self.actor, self.critic, self.policy = self._build_network()
        self.action_space = [i for i in range(self.n_actions)]

        # stage 0 = no sorting as a baseline
        # stage 1 = sorting of actions
        self.stage = 0

    def choose_action(self, observations):
        """ chooses actions based on observations
            each trace will be introduced to the network to find a fix
            sort actions afterwards for maximizing utility
            returns sorted list
        """
        actions = []
        for shop_name, components in observations.items():
            state = _encode_observations(components)[np.newaxis, :]
            if state.sum() > 0:  # skip shops without any failure
                probabilities = self.policy.predict(state)[0]
                action = np.random.choice(self.action_space, p=probabilities)
                decoded_action = _decoded_action(action, observations)
                actions.append({'shop': shop_name, 'component': decoded_action})
        if self.stage == 0:
            return actions
        else:
            return self._sort_actions(actions)

    def learn(self, states, actions, reward, states_, dones):
        """ network learns to improve """
        actions = {action['shop']: action['component'] for action in actions.values()}
        for shop_name, action in actions.items():
            state = _encode_observations(states[shop_name])[np.newaxis, :]
            state_ = _encode_observations(states_[shop_name])[np.newaxis, :]

            critic_value_ = self.critic.predict(state_)
            critic_value = self.critic.predict(state)

            shop_reward = reward[0][shop_name]
            target = shop_reward + self.gamma * critic_value_ * (1 - int(dones))
            delta = target - critic_value

            _actions = np.zeros([1, self.n_actions])
            _actions[np.arange(1), self.action_space_inverted.index(action)] = 1.0

            self.actor.fit([state, delta], _actions, verbose=0)
            self.critic.fit(state, target, verbose=0)

    def _sort_actions(self, actions):
        """ sort actions with RidgeRegression
            returns sorted list
        """
        raise NotImplementedError

    def _build_network(self):
        if self.load_models_data is None:
            input = Input(shape=(self.input_dims,))
            delta = Input(shape=[1])
            dense1 = Dense(self.fc1_dims, activation='relu')(input)
            dense2 = Dense(self.fc2_dims, activation='relu')(dense1)

            probs = Dense(self.n_actions, activation='softmax')(dense2)
            values = Dense(1, activation='linear')(dense2)

            actor = Model(inputs=[input, delta], outputs=[probs])
            actor.compile(optimizer=Adam(lr=self.alpha), loss=_delta_custom_loss(delta))

            critic = Model(inputs=[input], outputs=[values])
            critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

            policy = Model(inputs=[input], outputs=[probs])

            return actor, critic, policy
        else:
            return self.load_models(self.load_models_data)

    def save(self, episode):
        self.actor.save(f"{self.base_model_dir}/{self.start_time}/agent_{self.index}/actor/episode_{episode}")
        self.critic.save(f"{self.base_model_dir}/{self.start_time}/agent_{self.index}/critic/episode_{episode}")
        self.policy.save(f"{self.base_model_dir}/{self.start_time}/agent_{self.index}/policy/episode_{episode}")

    def load_models(self, load_models_data):
        # recheck if model implementation has changed
        # maybe custom loss function is not working properly
        base_dir = f"{self.base_model_dir}/{load_models_data['start_time']}/agent_{self.index}"
        delta = Input(shape=[1])
        actor = tf.keras.models.load_model(
            f"{base_dir}/actor/episode_{load_models_data['episode']}",
            custom_objects={'custom_loss': _delta_custom_loss(delta)}
        )
        critic = tf.keras.models.load_model(f"{base_dir}/critic/episode_{load_models_data['episode']}")
        policy = tf.keras.models.load_model(f"{base_dir}/policy/episode_{load_models_data['episode']}")
        return actor, critic, policy
