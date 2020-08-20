# Unlike REINFORCE, we do not wait for end of episode to train.  Also, the advantage is now calculated
# from TD error, which are derived using critic network.

import gym
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from utils.visualizer import visualize
import random


class A2C:
    def __init__(self):
        self.gamma = 0.9   # Reward discount
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.01
        self.memory_size = 10000
        self.memory = []
        self.batch_size = 32

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
        layer = Dense(1)(layer)
        output = Activation('linear')(layer)
        model = Model(input, output)
        adam = Adam(lr=self.critic_learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model

    # Select action based on the output of policy network
    def select_action(self, state, model):
        state = np.reshape(state, (1, self.num_states))
        policy = model.predict(state).ravel()

        return np.random.choice(self.num_actions, 1, p=policy)[0]

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

        actor = self.build_actor()
        critic = self.build_critic()

        max_episode = 100000
        max_step = 10000
        slide_window = 50

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
                    visualize(episode_reward, episode_count, slide_window, "A2C_Offpolicy.png")
                    break

                state = next_state
                current_step += 1

if __name__ == '__main__':
    agent = A2C()
    agent.train()
