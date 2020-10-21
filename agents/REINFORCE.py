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


class REINFORCE():
    def __init__(self, config, env):
        self.lr = config["lr"]
        self.gamma = config["gamma"]   # Reward discount
        self.layers = config["layers"]
        self.memory = []
        self.result_path = config["result_path"]
        self.env = env
        self.render_environment = config["render_environment"]
        self.max_episode = config["max_episode"]
        self.max_step = config["max_step"]
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0]
        self.agent = self.build_agent()
        self.logger = Logger(config["slide_window"])

        # Storage for states, actions and rewards in one episode
        self.states, self.actions, self.rewards = [], [], []

    def build_agent(self):
        input = Input(shape=(self.num_states,))
        layer = input
        for num_nodes in self.layers:
            layer = Dense(num_nodes, activation='relu')(layer)
        logit = Dense(self.num_actions)(layer)
        output = Activation('softmax')(logit)
        model = Model(input, output)
        adam = Adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy', optimizer=adam)

        return model

    # Select action based on the output of policy network
    def select_action(self, state):
        state = np.reshape(state, (1, self.num_states))
        policy = self.agent.predict(state).ravel()

        return np.random.choice(self.num_actions, 1, p=policy)[0]

    # Uses Monte Carlo for reward calculation
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        return discounted_rewards

    # Save <s, a ,r> of each step
    def store_memory(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def train(self):
        t = trange(self.max_episode)
        for episode_count in t:
            state = self.env.reset()
            current_step = 0

            while True:
                if self.render_environment:
                    self.env.render()

                # Select action, observe reward and state, then store them
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.store_memory(state, action, reward)

                # Start training when an episode finishes
                if done or current_step > self.max_step:
                    episode_reward = np.sum(self.rewards)

                    # Train policy network every episode
                    episode_length = len(self.states)
                    discounted_rewards = self.discount_rewards(self.rewards)
                    advantages = np.zeros((episode_length, self.num_actions))

                    for i in range(episode_length):
                        advantages[i][self.actions[i]] = discounted_rewards[i]

                    # history = model.fit(np.array(self.states), advantages, nb_epoch=1, verbose=0, batch_size=10)
                    self.agent.train_on_batch(np.array(self.states), advantages)

                    # For logging and visualizing data
                    self.logger.log_history(episode_reward, episode_count)
                    self.logger.show_progress(t, episode_reward, episode_count)
                    if episode_count % self.logger.slide_window == 0:
                        visualize(self.logger.rewards,
                                  self.logger.running_rewards,
                                  self.logger.episode_counts,
                                  os.path.join(self.result_path, "REINFORCE.png"))

                    # Clear memory after episode finishes
                    self.states, self.actions, self.rewards = [], [], []

                    break

                state = next_state
                current_step += 1

if __name__ == '__main__':
    agent = REINFORCE()
    agent.train()