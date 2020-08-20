# Everything is the same as A2C, except the action is now calculated sampled from a normal
# distribution with mean and standard deviation (sqrt(exp(log_var))).

import gym
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from utils.visualizer import visualize


class A2C:
    def __init__(self):
        self.gamma = 0.9   # Reward discount
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.01
        self.memory = []
        self.batch_size = 32

    def build_actor(self):
        input = Input(shape=(self.num_states,))
        advantage = Input(shape=(self.num_actions,), name="advantage")
        action = Input(shape=(self.num_actions,), name="action")

        layer = Dense(50, activation='relu')(input)
        layer = Dense(256, activation='relu')(layer)
        mu = Dense(self.num_actions)(layer)
        log_var = Dense(self.num_actions)(layer)
        mu = Activation("tanh")(mu)
        log_var = Activation("tanh")(log_var)
        model = Model([input, advantage, action], [mu, log_var])
        adam = Adam(lr=self.actor_learning_rate)

        def custom_loss(y_true, y_pred):
            logprob = -0.5 * (log_var + K.square(action - mu) / K.exp(log_var))
            loss = -logprob * advantage
            #action_logprobs += -0.5 * K.square(self.action - mu) / K.exp(log_var)
            return loss

        model.compile(loss=custom_loss, optimizer=adam)

        return model

    def build_critic(self):
        input = Input(shape=(self.num_states,))
        layer = Dense(50, activation='relu')(input)
        layer = Dense(256, activation='relu')(layer)
        logit = Dense(1)(layer)
        output = Activation('linear')(logit)
        model = Model(input, output)
        adam = Adam(lr=self.critic_learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model

    # Select action based on the output of policy network
    def select_action(self, state, model):
        state = np.reshape(state, (1, self.num_states))
        mu, log_var = model.predict([state, self.dummy_input, self.dummy_input])
        mu, log_var = mu[0], log_var[0]
        var = np.exp(log_var)
        action = np.random.normal(loc=mu, scale=np.sqrt(var))

        return action

    def store_memory(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def train(self):
        # Setup environment first
        env = gym.make('MountainCarContinuous-v0')
        env.seed(1)
        env = env.unwrapped

        self.num_actions = env.action_space.shape[0]
        self.num_states = env.observation_space.shape[0]

        # Dummy inputs used during prediction
        self.dummy_input = np.zeros((1, self.num_actions))
        self.dummy_batch_input = np.zeros((self.batch_size, self.num_actions))

        actor = self.build_actor()
        critic = self.build_critic()

        max_episode = 100000
        max_step = 10000
        slide_window = 20

        for episode_count in range(max_episode):
            state = env.reset()
            current_step = 0
            episode_reward = 0

            while True:
                #env.render()

                # Actor select action, observe reward and state, then store them
                action = self.select_action(state, actor)
                action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
                next_state, reward, done, info = env.step(action)

                # Store transition
                episode_reward += reward
                self.store_memory(state, action, reward, next_state, done)

                # Batch train once enough samples in memory
                if len(self.memory) >= self.batch_size:
                    self.memory = list(zip(*self.memory))
                    states, actions, rewards, next_states, dones = self.memory

                    # Calculate advantage
                    batch_values = critic.predict_on_batch(np.array(states)).ravel()
                    batch_next_values = critic.predict_on_batch(np.array(next_states)).ravel()
                    td_targets = rewards + self.gamma * (1 - np.array(dones)) * batch_next_values
                    td_errors = td_targets - batch_values
                    advantages = np.zeros((self.batch_size, self.num_actions))
                    batch_actions = np.zeros((self.batch_size, self.num_actions))
                    for i in range(self.batch_size):
                        for j in range(self.num_actions):
                            advantages[i][j] = td_errors[i]
                            batch_actions[i][j] = actions[i]

                    # Train critic
                    critic.train_on_batch(np.array(states), np.array(td_targets))

                    # Train actor
                    actor.train_on_batch([np.array(states), advantages, batch_actions], [self.dummy_batch_input, self.dummy_batch_input])

                    self.memory = []

                # For logging data
                if done or current_step > max_step:
                    visualize(episode_reward, episode_count, slide_window, "A2C_Continuous.png")
                    break

                state = next_state
                current_step += 1

if __name__ == '__main__':
    agent = A2C()
    agent.train()