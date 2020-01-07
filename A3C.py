# TODO: unfinished
# A3C

import gym
import numpy as np
from keras.layers import Dense, Input, Activation
from keras import Model
from keras.optimizers import Adam
from util import visualization
from threading import Thread, Lock
import threading
import tqdm
from keras.utils import to_categorical
from utils.networks import tfSummary


class A2C:
    def __init__(self):
        self.gamma = 0.9   # Reward discount
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.01
        self.memory = []
        self.batch_size = 32
        self.render = False

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
        self.memory.append([state, action, reward, next_state, done])

    # Function for worker threads to run training.
    def worker_train(self, agent, Nmax, env, action_dim, f, summary_writer, tqdm, render):
        lock = Lock()

        global episode
        while episode < Nmax:
            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()
            actions, states, rewards = [], [], []
            while not done and episode < Nmax:
                if render:
                    with lock: env.render()
                # Actor picks an action (following the policy)
                a = agent.policy_action(np.expand_dims(old_state, axis=0))
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a)
                # Memorize (s, a, r) for training
                actions.append(to_categorical(a, action_dim))
                rewards.append(r)
                states.append(old_state)
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1
                # Asynchronous training
                if (time % f == 0 or done):
                    lock.acquire()
                    agent.train_models(states, actions, rewards, done)
                    lock.release()
                    actions, states, rewards = [], [], []

            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=episode)
            summary_writer.flush()
            # Update episode count
            with lock:
                tqdm.set_description("Score: " + str(cumul_reward))
                tqdm.update(1)
                if episode < Nmax:
                    episode += 1

    def train(self):
        # Setup environment first
        env = gym.make('CartPole-v1')
        env.seed(1)
        env = env.unwrapped

        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0]

        global_actor = self.build_actor()
        global_critic = self.build_critic()

        max_episode = 100000
        max_step = 10000
        slide_window = 50

        for episode_count in range(max_episode):
            state = env.reset()
            current_step = 0
            episode_reward = 0

            while True:
                if self.render:
                    env.render()

                # Actor select action, observe reward and state, then store them
                action = self.select_action(state, global_actor)
                next_state, reward, done, info = env.step(action)

                # Store transition
                episode_reward += reward
                self.store_memory(state, action, reward, next_state, done)

                # Batch train once enough samples in memory
                if len(self.memory) >= self.batch_size:
                    self.memory = list(zip(*self.memory))
                    states, actions, rewards, next_states, dones = self.memory

                    # Calculate advantage
                    batch_values = global_critic.predict_on_batch(np.array(states)).ravel()
                    batch_next_values = global_critic.predict_on_batch(np.array(next_states)).ravel()
                    td_targets = rewards + self.gamma * (1 - np.array(dones)) * batch_next_values
                    td_errors = td_targets - batch_values
                    advantages = np.zeros((self.batch_size, self.num_actions))
                    for i in range(self.batch_size):
                        advantages[i][actions[i]] = td_errors[i]

                    # Train global_critic
                    global_critic.train_on_batch(np.array(states), np.array(td_targets))

                    # Train global_actor
                    global_actor.train_on_batch(np.array(states), np.array(advantages))

                    self.memory = []

                # For logging data
                if done or current_step > max_step:
                    visualization(episode_reward, episode_count, slide_window, "A2C.png")
                    break

                state = next_state
                current_step += 1

if __name__ == '__main__':
    agent = A2C()
    agent.train()
