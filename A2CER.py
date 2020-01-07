# Unlike REINFORCE, we do not wait for end of episode to train.  Also, the advantage is now calculated
# from TD error, which are derived using critic network.

import gym
import numpy as np
from keras.layers import Dense, Input, Activation
from keras import Model
from keras.optimizers import Adam
from util import visualization
import keras.backend as K
import random
from itertools import count


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


class A2C:
    def __init__(self):
        self.gamma = 0.9   # Reward discount
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.01
        self.memory_size = 10000
        self.memory = SumTree(self.memory_size)
        self.batch_size = 32
        self.replay_frequency = 10
        self.counter = count()  # In case multiple memories with same priority
        self.alpha = 0.6    # Used during priority calculation
        self.beta = 0.4  # Used during importance sampling
        self.beta_increase = 0.001
        self.beta_max = 1
        self.constant = 1e-10   # In case priority is 0

    def build_actor(self):
        input = Input(shape=(self.num_states,))
        layer = Dense(50, activation='relu')(input)
        layer = Dense(256, activation='relu')(layer)
        logit = Dense(self.num_actions)(layer)
        output = Activation('softmax')(logit)
        model = Model(input, output)
        adam = Adam(lr=self.actor_learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam)

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
        policy = model.predict(state).ravel()

        return np.random.choice(self.num_actions, 1, p=policy)[0]

    def store_memory(self, priority, state, action, reward, next_state, done):
        self.memory.add(priority, [state, action, reward, next_state, done])

    def train(self):
        # Setup environment first
        env = gym.make('CartPole-v1')
        env.seed(1)
        env = env.unwrapped

        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0]

        actor = self.build_actor()
        critic = self.build_critic()

        max_episode = 100000
        max_step = 10000
        slide_window = 50

        # Populate memory first
        state = env.reset()
        print("Warming up...")
        for i in range(self.batch_size):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            self.store_memory(1, state, action, reward, next_state, done)   # Priority=1 for these transitions
            if done:
                state = env.reset()
        print("Warm up complete.")

        for episode_count in range(max_episode):
            state = env.reset()
            current_step = 0
            episode_reward = 0

            while True:
                #env.render()

                # Actor select action, observe reward and state, then store them
                action = self.select_action(state, actor)
                next_state, reward, done, info = env.step(action)

                # Store transition
                episode_reward += reward
                self.store_memory(state, action, reward, next_state, done)

                # Sample minibatch from memory
                minibatch = random.sample(self.memory, self.batch_size)

                # Transform the minibatch for processing
                minibatch = list(zip(*minibatch))

                # Calculate advantage
                states, actions, rewards, next_states, dones = minibatch
                batch_values = critic.predict_on_batch(np.array(states)).ravel()
                batch_next_values = critic.predict_on_batch(np.array(next_states)).ravel()
                td_targets = rewards + self.gamma * (1 - np.array(dones)) * batch_next_values
                td_errors = td_targets - batch_values
                advantages = np.zeros((self.batch_size, self.num_actions))
                for i in range(self.batch_size):
                    advantages[i][actions[i]] = td_errors[i]

                # Train critic
                critic.train_on_batch(np.array(states), np.array(td_targets))

                # Train actor
                actor.train_on_batch(np.array(states), np.array(advantages))

                # For logging data
                if done or current_step > max_step:
                    visualization(episode_reward, episode_count, slide_window, "A2C_Offpolicy.png")
                    break

                state = next_state
                current_step += 1

if __name__ == '__main__':
    agent = A2C()
    agent.train()
