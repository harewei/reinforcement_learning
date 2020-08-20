import gym
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from utils.visualizer import visualize


class REINFORCE():
    def __init__(self):
        self.num_layers = 2
        self.num_nodes = 10
        self.gamma = 0.95   # reward discount
        self.learning_rate = 0.01

        # Storage for states, actions and rewards in one episode
        self.states, self.actions, self.rewards = [], [], []

    def build_model(self):
        input = Input(shape=(self.num_states,))
        layer = Dense(self.num_nodes, activation='relu')(input)
        logit = Dense(self.num_actions)(layer)
        output = Activation('softmax')(logit)
        model = Model(input, output)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam)

        return model

    # Select action based on the output of policy network
    def select_action(self, state, model):
        state = np.reshape(state, (1, self.num_states))
        policy = model.predict(state).ravel()

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

    # Train policy network every episode
    def train_model(self, model):
        episode_length = len(self.states)
        discounted_rewards = self.discount_rewards(self.rewards)
        advantages = np.zeros((episode_length, self.num_actions))

        for i in range(episode_length):
            advantages[i][self.actions[i]] = discounted_rewards[i]

        #history = model.fit(np.array(self.states), advantages, nb_epoch=1, verbose=0, batch_size=10)
        history = model.train_on_batch(np.array(self.states), advantages)

        return history

    def train(self):
        # Setup environment first
        env = gym.make('CartPole-v1')
        env.seed(1)
        env = env.unwrapped

        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0]

        model = self.build_model()

        max_episode = 100000
        max_step = 10000
        slide_window = 100

        for episode_count in range(max_episode):
            state = env.reset()
            current_step = 0

            while True:
                #env.render()

                # Select action, observe reward and state, then store them
                action = self.select_action(state, model)
                next_state, reward, done, info = env.step(action)
                self.store_memory(state, action, reward)

                # Start training when an episode finishes
                if done or current_step > max_step:
                    episode_reward = np.sum(self.rewards)

                    # Training
                    history = self.train_model(model)

                    # For logging data
                    if done or current_step > max_step:
                        visualize(episode_reward, episode_count, slide_window, "REINFORCE.png")

                    # Clear memory after episode finishes
                    self.states, self.actions, self.rewards = [], [], []

                    break

                state = next_state
                current_step += 1

if __name__ == '__main__':
    agent = REINFORCE()
    agent.train()