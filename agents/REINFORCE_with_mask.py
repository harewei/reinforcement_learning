import gym
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, InputLayer, Input, Activation, Lambda, multiply
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = True  # rendering wastes time

np.random.seed(1)
tf.set_random_seed(1)

class REINFORCE_Agent():
    def __init__(self):
        self.num_states = 135
        self.num_actions = 104
        self.num_layers = 2
        self.num_nodes = 10
        self.discount_factor = 0.95
        self.learning_rate = 0.01

        # Storage for states, actions and rewards in one episode
        self.states, self.actions, self.rewards, self.action_masks = [], [], [], []
        self.policies = []
        self.uncertain_moves = 0

    def build_model(self):
        input = Input(shape=(self.num_states,))
        mask = Input(shape=(self.num_actions,))
        #layer = Dense(self.num_nodes, activation='tanh')(input)
        layer = Dense(self.num_nodes, activation='relu')(input)
        logit = Dense(self.num_actions)(layer)
        masked_logit = Lambda(lambda x: x + mask)(logit)
        output = Activation('softmax')(logit)#(masked_logit)
        model = Model([input, mask], output)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam)

        return model

    # Select action based on the output of policy network
    def select_action(self, state, possible_actions_list, model):
        state = np.reshape(state, (1, self.num_states))
        # Create mask so that agent doesn't choose from invalid actions
        actions_mask = np.ones(self.num_actions)
        actions_mask = actions_mask*-np.inf
        for i in range(len(possible_actions_list)):
            actions_mask[possible_actions_list[i]] = 0
        self.action_masks.append(actions_mask)
        actions_mask = np.reshape(actions_mask, (1, self.num_actions))
        policy = model.predict([state, actions_mask]).ravel()
        if np.argmax(policy) not in possible_actions_list:
            self.uncertain_moves += 1

        return np.random.choice(self.num_actions, 1, p=policy)[0]

    # Uses Monte Carlo for reward calculation
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
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

        #history = model.fit([np.array(self.states), np.array(self.action_masks)], advantages, nb_epoch=1, verbose=0, batch_size=10)
        history = model.train_on_batch([np.array(self.states), np.array(self.action_masks)], advantages)

        return history

    def train(self):
        # Setup environment first
        env = gym.make('CartPole-v1') #gym.make('MountainCar-v0')
        env.seed(1)  # reproducible, general Policy gradient has high variance
        env = env.unwrapped

        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0]

        model = self.build_model()

        max_episode = 100000
        uncertain_history = []
        reward_history = []
        step_history = []
        slide_window = 50
        episode_history = []

        for episode_count in range(max_episode):
            state = env.reset()

            while True:
                #env.render()

                # Select action, observe reward and state, then store them
                possible_actions_list = np.arange(0, self.num_actions)
                action = self.select_action(state, possible_actions_list, model)
                next_state, reward, done, info = env.step(action)
                self.store_memory(state, action, reward)

                # Start training when an episode finishes
                if done:
                    episode_reward = np.sum(self.rewards)
                    episode_steps = len(self.states)

                    # Training
                    history = self.train_model(model)

                    # For logging data
                    reward_history.append(episode_reward)
                    episode_history.append(episode_count)
                    #uncertain_history.append(self.uncertain_moves / episode_steps * 100)
                    plt.figure(1)
                    plt.title("Episode Reward")
                    plt.xlabel('Episode')
                    plt.ylabel('Reward')
                    plt.plot(episode_history, reward_history)
                    if episode_reward > 2000:
                        plt.savefig("REINFORCE.png")

                    print(episode_reward)

                    # Clear memory after episode finishes
                    self.states, self.actions, self.rewards, self.action_masks = [], [], [], []
                    self.policies = []
                    self.uncertain_moves = 0

                    break

                state = next_state

if __name__ == '__main__':
    agent = REINFORCE_Agent()
    agent.train()