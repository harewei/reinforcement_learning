# Double DQN.  Compared to DQN, it uses q_network rather than target_q_network
# when selecting next action when extracting next q value.

import gym
import numpy as np
import random
from keras.layers import Dense, Input, Activation
from keras import Model
from keras.optimizers import Adam
from util import visualization


class DDQN():
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
    def store_memory(self, state, action, reward, next_state, done):
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

        self.memory.append([state, action, reward, next_state, done])

    def train(self):
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
        slide_window = 100

        # Populate memory first
        state = env.reset()
        print("Warming up...")
        while len(self.memory) < self.batch_size:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            self.store_memory(state, action, reward, next_state, done)
            if done:
                state = env.reset()
        print("Warm up complete.")

        for episode_count in range(max_episode):
            state = env.reset()
            current_step = 0
            episode_reward = 0

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
                next_actions = np.argmax(q_network.predict_on_batch(np.array(next_states)), axis=1) # Main difference between DDQN and DQN
                td_targets = batch_q_values.copy()
                for i in range(self.batch_size):
                    td_targets[i][actions[i]] = rewards[i] + self.gamma * (1 - dones[i]) * batch_next_q_values[i][next_actions[i]]

                # Train network
                q_network.train_on_batch(np.array(states), np.array(td_targets))

                # Hard copy q_network to target_q_network
                if done or current_step % self.update_frequency is 0:
                    target_q_network.set_weights(q_network.get_weights())

                # For logging data
                if done or current_step > max_step:
                    visualization(episode_reward, episode_count, slide_window, "DDQN.png")
                    break

                state = next_state
                current_step += 1

if __name__ == '__main__':
    agent = DDQN()
    agent.train()