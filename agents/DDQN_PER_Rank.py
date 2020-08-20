# TODO: Unfinished algorithm
# Double DQN with Experience Replay using Rank method (heapq).

import gym
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import heapq
from itertools import count

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = True  # rendering wastes time

class DDQN_PER():
    def __init__(self):
        self.gamma = 0.95   # reward discount
        self.learning_rate = 0.001
        self.memory_size = 10000
        self.epsilon = 0.8  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.memory = []
        self.batch_size = 32
        self.rewards = []
        self.update_frequency = 50
        self.counter = count()  # In case multiple memories with same priority
        self.replay_period = 50
        self.edge = 1e-10   # In case priority is 0

    def build_network(self):
        input = Input(shape=(self.num_states,))
        layer = Dense(24, activation='relu')(input)
        layer = Dense(self.num_actions)(layer)
        output = Activation('linear')(layer)
        model = Model(input, output)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model

    # Save <s, a ,r, s'> of each step
    def store_memory(self, priority, state, action, reward, next_state, done):
        if len(self.memory) > self.memory_size:
            heapq.heappop(self.memory)

        heapq.heappush(self.memory, [priority, next(self.counter), state, action, reward, next_state, done])

    def train(self):
        # a = [[1, 2], [30]]
        # b = [[2], [20]]
        # c = [[1, 1], [40]]
        # h = []
        # heapq.heapify(h)
        # heapq.heappush(h, c)
        # heapq.heappush(h, a)
        # heapq.heappush(h, b)
        # print(h)
        # print(heapq.heappop(h))
        # print(h)

        # Setup environment first
        env = gym.make('CartPole-v1')
        env.seed(1)
        env = env.unwrapped

        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0]

        # Initialize q_network and target_q_network
        q_network = self.build_network()
        target_q_network = self.build_network()
        target_q_network.set_weights(q_network.get_weights())

        max_episode = 100000
        max_step = 10000
        reward_history = []
        episode_history = []

        # Populate memory first
        state = env.reset()
        print("Warming up...")
        while len(self.memory) < self.batch_size:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            self.store_memory(1, state, action, reward, next_state, done)
            if done:
                state = env.reset()
        print("Warm up complete.")

        for episode_count in range(max_episode):
            state = env.reset()
            current_step = 0

            while True:
                #env.render()

                # Network predict
                q_values = q_network.predict(np.reshape(state, (1, self.num_states))).ravel()

                # Decide if exploring or not
                if np.random.rand() >= self.epsilon:
                    action = np.argmax(q_values)
                else:
                    action = random.randrange(self.num_actions)

                # Perform action
                next_state, reward, done, info = env.step(action)

                # Calculate priority
                next_q_values = target_q_network.predict(np.reshape(next_state, (1, self.num_states))).ravel()
                next_action = q_network.predict(np.reshape(next_state, (1, self.num_states))).ravel()
                td_error = reward + self.gamma * (1 - done) * next_q_values[next_action] - q_values[action]
                priority = abs(td_error) + self.edge

                # Store transition
                self.store_memory(priority, state, action, reward, next_state, done)
                self.rewards.append(reward)

                # Decrease exploration
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                # Sample minibatch from memory based on their priority
                priorities = self.memory
                minibatch = random.sample(self.memory, self.batch_size)

                # Transform the minibatch for processing
                minibatch = list(zip(*minibatch))

                # Calculate all td_targets for current minibatch
                priorities, _, states, actions, rewards, next_states, dones = minibatch
                batch_q_values = q_network.predict_on_batch(np.array(states))
                batch_next_q_values = target_q_network.predict_on_batch(np.array(next_states))
                next_actions = np.argmax(q_network.predict_on_batch(np.array(next_states)), axis=1)
                td_targets = batch_q_values.copy
                for i in range(self.batch_size):
                    td_targets[i][actions[i]] = rewards[i] + self.gamma * (1 - dones[i]) * batch_next_q_values[i][next_actions[i]]

                # Train network
                q_network.train_on_batch(np.array(states), np.array(td_targets))

                # Soft copy q_network to target_q_network
                if done or current_step % self.update_frequency is 0:
                    target_q_network.set_weights(q_network.get_weights())

                # For logging data
                if done or current_step > max_step:
                    episode_reward = np.sum(self.rewards)
                    reward_history.append(episode_reward)
                    episode_history.append(episode_count)
                    plt.figure(1)
                    plt.title("Episode Reward")
                    plt.xlabel('Episode')
                    plt.ylabel('Reward')
                    plt.plot(episode_history, reward_history)
                    #if episode_reward > 2000:
                    if episode_count % 100 is 0:
                        plt.savefig("DDQN_PER.png")

                    print("episode: {}, reward: {}".format(episode_count, episode_reward))
                    self.rewards = []

                    break

                state = next_state
                current_step += 1

if __name__ == '__main__':
    agent = DDQN_PER()
    agent.train()