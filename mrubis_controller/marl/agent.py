from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam

import numpy as np


class Agent:
    def __init__(self, shops):
        self.shops = shops
        self.gamma = 0.99
        self.alpha = 0.00001
        self.beta = 0.00005
        self.input_dims = 8
        self.fc1_dims = 1024
        self.fc2_dims = 512
        self.n_actions = 18

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
        for observation in observations:
            state = observation[np.newaxis, :]
            probabilities = self.policy.predict(state)[0]
            action = np.random.choice(self.action_space, p=probabilities)
            actions.append(action)
        if self.stage == 0:
            return actions
        else:
            return self._sort_actions(actions)

    def learn(self, state, action, reward, state_, done):
        """ network learns to improve """
        state = state[np.newaxis, :]
        state_ = state_[np.newaxis, :]

        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma * critic_value_ * (1 - int(done))
        delta = target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1.0

        self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target, verbose=0)

    def _sort_actions(self, actions):
        """ sort actions with RidgeRegression
            returns sorted list
        """
        raise NotImplementedError

    def _build_network(self):
        input = Input(shape=(self.input_dims,))
        delta = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)

        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1 - 1e-8)
            log_like = y_true * K.log(out)

            return K.sum(-log_like * delta)

        actor = Model(inputs=[input, delta], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        critic = Model(inputs=[input], outputs=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        policy = Model(inputs=[input], outputs=[probs])

        return actor, critic, policy
