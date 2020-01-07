# Double DQN.  Compared to DQN, it uses q_network rather than target_q_network
# when selecting next action when extracting next q value.

import gym
import numpy as np
import random
from keras.layers import Dense, Input, Activation
from keras import Model
from keras.optimizers import Adam
from util import visualization


class C51_DDQN():
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
        self.num_atoms = 51
        self.v_max = 1000 # Max possible score for Defend the center is 26 - 0.1*26 = 23.4
        self.v_min = 0

    def build_network(self):
        input = Input(shape=(self.num_states,))
        layer = Dense(24, activation='relu')(input)
        distribution_list = []
        # C51 atoms
        for i in range(self.num_actions):
            distribution_list.append(Dense(self.num_atoms, activation='softmax')(layer))
        model = Model(input, distribution_list)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam)

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

        # Initialize z_network and target_z_network
        z_network = self.build_network()
        target_z_network = self.build_network()
        target_z_network.set_weights(z_network.get_weights())

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
                z_values = z_network.predict(np.reshape(state, (1, self.num_states)))
                q_values = np.sum(np.multiply(np.squeeze(z_values), np.squeeze(np.array(z_values))), axis=1)
                #q_values = q_values.reshape((self.batch_size, self.num_actions), order='F')

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
                batch_z_values = z_network.predict_on_batch(np.array(states))
                batch_next_z_values = target_z_network.predict_on_batch(np.array(next_states))
                batch_z_values = np.swapaxes(batch_z_values, 0, 1)  # I like dealing with batch size at axis 0
                batch_next_z_values = np.swapaxes(batch_next_z_values, 0, 1)
                batch_q_values = np.sum(np.multiply(np.array(batch_z_values), np.array(batch_z_values)), axis=2)
                next_actions = np.argmax(batch_q_values, axis=1)
                exit()
                td_targets = batch_z_values.copy()
                for i in range(self.batch_size):
                    td_targets[i][actions[i]] = rewards[i] + self.gamma * (1 - dones[i]) * batch_next_z_values[i][next_actions[i]]

                # Train network
                z_network.train_on_batch(np.array(states), np.array(td_targets))

                # Hard copy z_network to target_z_network
                if done or current_step % self.update_frequency is 0:
                    target_z_network.set_weights(z_network.get_weights())

                # For logging data
                if done or current_step > max_step:
                    visualization(episode_reward, episode_count, slide_window, "DDQN.png")
                    break

                state = next_state
                current_step += 1

if __name__ == '__main__':
    agent = C51_DDQN()
    agent.train()