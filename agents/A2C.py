# Unlike REINFORCE, we do not wait for end of episode to train.  Also, the advantage is now calculated
# from TD error, which are derived using critic network.

import os
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tqdm import trange
from utils.logger import Logger
from utils.visualizer import visualize


class A2C:
    def __init__(self, config, env):
        self.gamma = config["gamma"]   # Reward discount
        self.actor_lr = config["actor_lr"]
        self.critic_lr = config["critic_lr"]
        self.actor_layers = config["actor_layers"]
        self.critic_layers = config["critic_layers"]
        self.memory = []
        self.result_path = config["result_path"]
        self.batch_size = config["batch_size"]
        self.env = env
        self.render_environment = config["render_environment"]
        self.max_episode = config["max_episode"]
        self.max_step = config["max_step"]
        self.max_episode = config["max_episode"]
        self.max_step = config["max_step"]
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0]
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.logger = Logger(config["slide_window"])

    def build_actor(self):
        input = Input(shape=(self.num_states,))
        layer = input
        for num_nodes in self.actor_layers:
            layer = Dense(num_nodes, activation='relu')(layer)
        logit = Dense(self.num_actions)(layer)
        output = Activation('softmax')(logit)
        model = Model(input, output)
        adam = Adam(lr=self.actor_lr)
        model.compile(loss='categorical_crossentropy', optimizer=adam)

        return model

    def build_critic(self):
        input = Input(shape=(self.num_states,))
        layer = input
        for num_nodes in self.actor_layers:
            layer = Dense(num_nodes, activation='relu')(layer)
        layer = Dense(1)(layer)
        output = Activation('linear')(layer)
        model = Model(input, output)
        adam = Adam(lr=self.critic_lr)
        model.compile(loss='mse', optimizer=adam)

        return model

    # Select action based on the output of policy network
    def select_action(self, state, model):
        state = np.reshape(state, (1, self.num_states))
        policy = model.predict(state).ravel()

        return np.random.choice(self.num_actions, 1, p=policy)[0]

    def store_memory(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def train(self):
        t = trange(self.max_episode)
        for episode_count in t:
            state = self.env.reset()
            current_step = 0
            episode_reward = 0

            while True:
                if self.render_environment:
                    self.env.render()

                # Actor select action, observe reward and state, then store them
                action = self.select_action(state, self.actor)
                next_state, reward, done, info = self.env.step(action)

                # Store transition
                episode_reward += reward
                self.store_memory(state, action, reward, next_state, done)

                # Batch train once enough samples in memory
                if len(self.memory) >= self.batch_size:
                    self.memory = list(zip(*self.memory))
                    states, actions, rewards, next_states, dones = self.memory

                    # Calculate advantage
                    batch_values = self.critic.predict_on_batch(np.array(states)).ravel()
                    batch_next_values = self.critic.predict_on_batch(np.array(next_states)).ravel()
                    td_targets = rewards + self.gamma * (1 - np.array(dones)) * batch_next_values
                    td_errors = td_targets - batch_values
                    advantages = np.zeros((self.batch_size, self.num_actions))
                    for i in range(self.batch_size):
                        advantages[i][actions[i]] = td_errors[i]

                    # Train critic
                    self.critic.train_on_batch(np.array(states), np.array(td_targets))

                    # Train actor
                    self.actor.train_on_batch(np.array(states), np.array(advantages))

                    self.memory = []

                # For logging and visualizing data
                if done or current_step > self.max_step:
                    self.logger.log_history(episode_reward, episode_count)
                    self.logger.show_progress(t, episode_reward, episode_count)
                    if episode_count % self.logger.slide_window == 0:
                        visualize(self.logger.rewards,
                                  self.logger.running_rewards,
                                  self.logger.episode_counts,
                                  os.path.join(self.result_path, "A2C.png"))
                    break

                state = next_state
                current_step += 1

