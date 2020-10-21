# Note this is the newer version (2015) of DQN, which uses an extra target network compared to
# the old one.

import numpy as np
import os
import random

from agent import Agent
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tqdm import trange
from utils.logger import Logger
from utils.visualizer import visualize


class DQN(Agent):
    def __init__(self, config, env):
        self.gamma = config["gamma"]   # reward discount
        self.learning_rate = config["learning_rate"]
        self.memory_size = config["memory_size"]
        self.epsilon = config["epsilon"]  # Exploration rate
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]
        self.batch_size = config["batch_size"]
        self.update_frequency = config["update_frequency"]
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0]
        self.max_episode = config["max_episode"]
        self.max_step = config["max_step"]
        self.render_environment = config["render_environment"]
        self.result_path = config["result_path"]
        self.memory = []
        self.env = env
        self.build_agent()
        self.logger = Logger(config["slide_window"])

    def build_agent(self):
        input = Input(shape=(self.num_states,))
        layer = Dense(24, activation='relu')(input)
        layer = Dense(self.num_actions)(layer)
        output = Activation('linear')(layer)
        model = Model(input, output)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model

    # Save <s, a ,r, s'> of each step
    def store_memory(self, state, action, reward, next_state, done):
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

        self.memory.append([state, action, reward, next_state, done])

    def train(self):
        # Initialize q_network and target_q_network
        q_network = self.build_agent()
        target_q_network = self.build_agent()
        target_q_network.set_weights(q_network.get_weights())

        # Populate memory first
        state = self.env.reset()
        print("Warming up...")
        while len(self.memory) < self.batch_size:
            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            self.store_memory(state, action, reward, next_state, done)
            if done:
                state = self.env.reset()
        print("Warm up complete.")

        t = trange(self.max_episode)
        for episode_count in t:
            state = self.env.reset()
            current_step = 0
            episode_reward = 0

            while True:
                if self.render_environment:
                    self.env.render()

                # Network predict
                q_values = q_network.predict(np.reshape(state, (1, self.num_states))).ravel()

                # Decide if exploring or not
                if np.random.rand() >= self.epsilon:
                    action = np.argmax(q_values)
                else:
                    action = random.randrange(self.num_actions)

                # Perform action
                next_state, reward, done, info = self.env.step(action)

                # Store transition
                episode_reward += reward
                self.store_memory(state, action, reward, next_state, done)

                # Decrease exploration
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                # Sample minibatch from memory
                minibatch = random.sample(self.memory, self.batch_size)

                # Transform the minibatch for processing
                minibatch = list(zip(*minibatch))

                # Calculate all td_targets for current minibatch
                states, actions, rewards, next_states, dones = minibatch
                batch_q_values = q_network.predict_on_batch(np.array(states))
                batch_next_q_values = target_q_network.predict_on_batch(np.array(next_states))
                max_next_q_values = np.max(batch_next_q_values, axis=1)
                td_targets = batch_q_values.copy()
                for i in range(self.batch_size):
                    td_targets[i][actions[i]] = rewards[i] + self.gamma * (1 - dones[i]) * max_next_q_values[i]

                # Train network
                q_network.train_on_batch(np.array(states), np.array(td_targets))

                # Copy q_network to target_q_network
                if done or current_step % self.update_frequency is 0:
                    target_q_network.set_weights(q_network.get_weights())

                # For logging and visualizing data
                if done or current_step > self.max_step:
                    self.logger.log_history(episode_reward, episode_count)
                    self.logger.show_progress(t, episode_reward, episode_count)
                    if episode_count % self.logger.slide_window == 0:
                        visualize(self.logger.rewards,
                                  self.logger.running_rewards,
                                  self.logger.episode_counts,
                                  os.path.join(self.result_path, "DQN.png"))
                    break

                state = next_state
                current_step += 1

if __name__ == '__main__':
    agent = DQN()
    agent.train()