# Double DQN with Experience Replay using Proportional method (sumtree).

import numpy as np
import os
import random

from agent import Agent
from itertools import count
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tqdm import trange
from utils.logger import Logger
from utils.visualizer import visualize

# Memory storage method when using proportional method
# Original Code by @jaara: https://github.com/jaara/AI-blog/blob/master/SumTree.py
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class DDQN_PER_Prop(Agent):
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
        self.slide_window = config["slide_window"]
        self.render_environment = config["render_environment"]
        self.result_path = config["result_path"]
        self.memory = SumTree(self.memory_size)
        self.counter = count()  # In case multiple memories with same priority
        self.replay_frequency = config["replay_frequency"]
        self.alpha = config["alpha"]    # Used during priority calculation
        self.beta = config["beta"]  # Used during importance sampling
        self.beta_increase = config["beta_increase"]
        self.beta_max = config["beta_max"]
        self.constant = 1e-10   # In case priority is 0
        self.max_episode = config["max_episode"]
        self.max_step = config["max_step"]
        self.env = env
        self.q_network = self.build_agent()
        self.target_q_network = self.build_agent()
        self.logger = Logger(config["slide_window"])

    def build_agent(self):
        input = Input(shape=(self.num_states,))
        layer = Dense(24, activation="relu")(input)
        layer = Dense(self.num_actions)(layer)
        output = Activation("linear")(layer)
        model = Model(input, output)
        adam = Adam(lr=self.learning_rate)

        def huber_loss(y_true, y_pred):
            return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1)
        # tf.keras.losses.Huber()

        model.compile(loss="mse", optimizer=adam)  # Not sure if using huber loss is good here

        return model

    # Save <s, a ,r, s"> of each step
    def store_memory(self, priority, state, action, reward, next_state, done):
        self.memory.add(priority, [state, action, reward, next_state, done])

    def train(self):
        # Initialize q_network and target_q_network
        self.target_q_network.set_weights(self.q_network.get_weights())

        # Populate memory first
        state = self.env.reset()
        print("Warming up...")
        for i in range(self.batch_size):
            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            self.store_memory(1, state, action, reward, next_state, done)   # Priority=1 for these transitions
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
                q_values = self.q_network.predict(np.reshape(state, (1, self.num_states))).ravel()

                # Decide if exploring or not
                if np.random.rand() >= self.epsilon:
                    action = np.argmax(q_values)
                else:
                    action = random.randrange(self.num_actions)

                # Perform action
                next_state, reward, done, info = self.env.step(action)

                # Calculate priority
                next_q_values = self.target_q_network.predict(np.reshape(next_state, (1, self.num_states))).ravel()
                next_action = np.argmax(self.q_network.predict(np.reshape(next_state, (1, self.num_states))).ravel())
                td_error = reward + self.gamma * (1 - done) * next_q_values[next_action] - q_values[action] # Note that the td_error is not ^2 like in DQN
                priority = (abs(td_error) + self.constant) ** self.alpha

                # Store transition
                episode_reward += reward
                self.store_memory(priority, state, action, reward, next_state, done)

                # Decrease exploration
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                if current_step % self.replay_frequency is 0:
                    # Sample minibatch from memory based on their priority
                    minibatch = []
                    ISWeights = []
                    min_prob = np.min(self.memory.tree[-self.memory.capacity] / self.memory.total())
                    T = self.memory.total() // self.batch_size
                    for i in range(self.batch_size):
                        a, b = T * i, T * (i + 1)
                        s = random.uniform(a, b)
                        idx, priority, data = self.memory.get(s)
                        probability = priority/self.memory.total()
                        ISWeights.append(np.power(probability/min_prob, -self.beta))
                        minibatch.append((*data, idx))
                    self.beta = np.min([self.beta_max, self.beta + self.beta_increase])

                    # Transform the minibatch for processing
                    minibatch = list(zip(*minibatch))

                    # Calculate all td_targets for current minibatch
                    states, actions, rewards, next_states, dones, indices = minibatch
                    batch_q_values = self.q_network.predict_on_batch(np.array(states))
                    batch_next_q_values = self.target_q_network.predict_on_batch(np.array(next_states))
                    next_actions = np.argmax(self.q_network.predict_on_batch(np.array(next_states)), axis=1)
                    td_targets = batch_q_values.copy()
                    for i in range(self.batch_size):
                        td_targets[i][actions[i]] = rewards[i] + self.gamma * (1 - dones[i]) * batch_next_q_values[i][next_actions[i]]
                        # Need to recalculate priorities for transitions in minibatch
                        priority = (abs(td_targets[i][actions[i]] - batch_q_values[i][actions[i]]) + self.constant) ** self.alpha
                        self.memory.update(indices[i], priority)

                    # Train network
                    self.q_network.train_on_batch(np.array(states), np.array(td_targets), np.array(ISWeights))

                    # Hard copy q_network to target_q_network
                    if done or current_step % self.update_frequency is 0:
                        self.target_q_network.set_weights(self.q_network.get_weights())

                # For logging and visualizing data
                if done or current_step > self.max_step:
                    self.logger.log_history(episode_reward, episode_count)
                    t.set_description("Episode: {}, Reward: {}".format(episode_count, episode_reward))
                    t.set_postfix(running_reward="{:.2f}".format(self.logger.running_rewards[-1]))
                    if episode_count % self.logger.slide_window == 0:
                        visualize(self.logger.rewards,
                                  self.logger.running_rewards,
                                  self.logger.episode_counts,
                                  os.path.join(self.result_path, "DDQN_PER_Prop.png"))
                    break

                state = next_state
                current_step += 1

if __name__ == "__main__":
    agent = DDQN_PER_Prop()
    agent.train()