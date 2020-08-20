# FIXME: continous not yet implemented
# FIXME: PPO poor performance already
# PPO algorithm is almost the same as A2C, except loss is clipped based on current and previous
# policy ratio.

import gym
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from utils.visualizer import visualize


class PPO:
    def __init__(self):
        self.gamma = 0.9   # Reward discount
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.01
        self.epsilon = 0.2  # Loss clipping
        self.alpha = 0.9  # Rate to update old_actor
        self.entropy = 0.1  # Entropy ratio
        self.memory = []
        self.batch_size = 32
        self.constant = 1e-10

    def build_actor(self):
        state = Input(shape=(self.num_states,), name="state")
        advantage = Input(shape=(self.num_actions,), name="advantage")
        old_policy = Input(shape=(self.num_actions,), name="old_policy")

        layer = Dense(50, activation='relu')(state)
        layer = Dense(256, activation='relu')(layer)
        logit = Dense(self.num_actions)(layer)
        output = Activation('softmax')(logit)
        model = Model([state, advantage, old_policy], output)
        adam = Adam(lr=self.actor_learning_rate)

        # Apply PPO clipping
        def ppo_loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_policy
            ratio = prob / (old_prob + self.constant)   # In case old_prob for the action was 0
            return -K.mean(K.minimum(ratio * advantage,
                                     K.clip(ratio,
                                            min_value=1 - self.epsilon,
                                            max_value=1 + self.epsilon) * advantage)
                           + self.entropy * (prob * K.log(prob + self.constant)))

        model.compile(loss=ppo_loss, optimizer=adam)

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
        policy = model.predict([state, self.dummy_input, self.dummy_input]).ravel()

        return np.random.choice(self.num_actions, 1, p=policy)[0]

    def store_memory(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def train(self):
        # Setup environment first
        env = gym.make('CartPole-v1')
        env.seed(1)
        env = env.unwrapped

        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0]

        # Dummy inputs used during prediction
        self.dummy_input = np.zeros((1, self.num_actions))
        self.dummy_batch_input = np.zeros((self.batch_size, self.num_actions))

        actor = self.build_actor()
        old_actor = self.build_actor()
        old_actor.set_weights(actor.get_weights())
        critic = self.build_critic()

        max_episode = 100000
        max_step = 10000
        slide_window = 100

        for episode_count in range(max_episode):
            state = env.reset()
            current_step = 0
            episode_reward = 0

            while True:
                #env.render()

                # Actor select action, observe reward and state
                action = self.select_action(state, actor)
                next_state, reward, done, info = env.step(action)

                # Store transition
                episode_reward += reward
                self.store_memory(state, action, reward, next_state, done)

                # Batch train once enough samples in memory
                if len(self.memory) >= self.batch_size:
                    self.memory = list(zip(*self.memory))
                    states, actions, rewards, next_states, dones = self.memory

                    # Calculate advantage
                    batch_values = critic.predict_on_batch(np.array(states)).ravel()
                    batch_next_values = critic.predict_on_batch(np.array(next_states)).ravel()
                    old_policies = old_actor.predict_on_batch([np.array(states), self.dummy_batch_input, self.dummy_batch_input])
                    td_targets = rewards + self.gamma * (1 - np.array(dones)) * batch_next_values
                    td_errors = td_targets - batch_values
                    advantages = np.zeros((self.batch_size, self.num_actions))
                    one_hot_actions = np.zeros((self.batch_size, self.num_actions))
                    for i in range(self.batch_size):
                        advantages[i][actions[i]] = td_errors[i]
                        one_hot_actions[i][actions[i]] = 1

                    # Train critic
                    critic.train_on_batch(np.array(states), np.array(td_targets))

                    # Train actor
                    actor.train_on_batch([np.array(states), advantages, old_policies], one_hot_actions)

                    # Update old_actor weights to use for next step
                    old_actor.set_weights(self.alpha*np.array(actor.get_weights()) +
                                          (1-self.alpha)*np.array(old_actor.get_weights()))

                    self.memory = []

                # For logging data
                if done or current_step > max_step:
                    visualize(episode_reward, episode_count, slide_window, "PPO.png")
                    break

                state = next_state
                current_step += 1

if __name__ == '__main__':
    agent = PPO()
    agent.train()