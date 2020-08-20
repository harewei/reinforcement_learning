# Almost the same as A2C, except uses target network for both actor and critic.
# The target network updates also uses soft copy instead of hard copy like DQN.
# The critic is now Q(s,a) instead of V(s).

import gym
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Input, Activation, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from utils.visualizer import visualize
from tensorflow.keras import backend as K
import random


class OrnsteinUhlenbeckProcess(object):
    """ Ornstein-Uhlenbeck Noise (original code by @slowbull)
    """
    def __init__(self, theta=0.15, mu=0, sigma=1, x0=0, dt=1e-2, n_steps_annealing=100, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size

    def generate(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x0 = x
        return x


class DDPG:
    def __init__(self):
        self.gamma = 0.9   # Reward discount
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.01
        self.memory_size = 10000
        self.memory = []
        self.batch_size = 32
        self.alpha = 0.9    # Target network update ratio

    def build_actor(self):
        input = Input(shape=(self.num_states,))
        layer = Dense(50, activation='relu')(input)
        layer = Dense(256, activation='relu')(layer)
        layer = Dense(self.num_actions)(layer)
        output = Activation('tanh')(layer)
        model = Model(input, output)
        adam = Adam(lr=self.actor_learning_rate)

        def ddpg_loss(y_true, y_pred):
            return - y_true * y_pred

        model.compile(loss=ddpg_loss, optimizer=adam)

        return model

    def build_critic(self):
        state_input = Input(shape=(self.num_states,))
        action_input = Input(shape=(self.num_actions,), name="action_input")
        s_layer = Dense(50, activation='relu')(state_input)
        a_layer = Dense(50, activation='linear')(action_input)
        layer = Concatenate(axis=-1)([s_layer, a_layer])
        layer = Dense(256, activation='relu')(layer)
        layer = Dense(1)(layer)
        output = Activation('linear')(layer)
        model = Model([state_input, action_input], output)
        adam = Adam(lr=self.critic_learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model

    # Select action based on the output of policy network
    def select_action(self, state, model, current_step, action_low, action_high):
        state = np.reshape(state, (1, self.num_states))
        action = model.predict(state).ravel()

        action = np.clip(action + self.noise.generate(current_step), action_low, action_high)

        return action

    def store_memory(self, state, action, reward, next_state, done):
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

        self.memory.append([state, action, reward, next_state, done])

    def train(self):
        # Setup environment first
        env = gym.make('MountainCarContinuous-v0')
        env.seed(1)
        env = env.unwrapped

        self.num_actions = env.action_space.shape[0]
        self.num_states = env.observation_space.shape[0]
        self.noise = OrnsteinUhlenbeckProcess(size=self.num_actions)

        actor = self.build_actor()
        target_actor = self.build_actor()
        target_actor.set_weights(actor.get_weights())

        critic = self.build_critic()
        target_critic = self.build_critic()
        target_critic.set_weights(critic.get_weights())

        q_gradients_connect = K.gradients(critic.output, critic.input[1])

        max_episode = 100000
        max_step = 1000
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

        sess = K.get_session()

        for episode_count in range(max_episode):
            state = env.reset()
            current_step = 0
            episode_reward = 0

            while True:
                #env.render()

                # Actor select action, observe reward and state, then store them
                action = self.select_action(state, actor, current_step, env.action_space.low[0],  env.action_space.high[0])
                next_state, reward, done, info = env.step(action)

                # Store transition
                episode_reward += reward
                self.store_memory(state, action, reward, next_state, done)

                # Sample minibatch from memory
                minibatch = random.sample(self.memory, self.batch_size)

                # Transform the minibatch for processing
                minibatch = list(zip(*minibatch))

                # Get td target
                states, actions, rewards, next_states, dones = minibatch
                next_actions = [self.select_action(next_state, target_actor, current_step, env.action_space.low[0],  env.action_space.high[0]) for next_state in next_states]
                batch_next_q_values = target_critic.predict_on_batch([np.array(next_states), np.array(next_actions)]).ravel()   # Action from policy is considered argmax of Q-value network
                td_targets = rewards + self.gamma * (1 - np.array(dones)) * batch_next_q_values

                # Get gradient of critic output wrt to the action input
                #tmp_actions = actor.predict_on_batch(np.array(states))
                #tmp_actions = np.clip(tmp_actions, env.action_space.low[0], env.action_space.high[0])
                #actions_for_gradients = critic.predict_on_batch([np.array(states), tmp_actions])
                actions_for_gradients = critic.predict_on_batch([np.array(states), np.array(actions)])

                q_gradients = sess.run(q_gradients_connect, feed_dict={critic.input[0]: np.array(states),
                                                                       critic.input[1]: actions_for_gradients})

                # Train critic
                critic.train_on_batch([np.array(states), np.array(actions)], np.array(td_targets))

                # Train actor
                actor.train_on_batch(np.array(states), q_gradients)

                # Soft update target actor and critic
                target_actor.set_weights((self.alpha * np.array(actor.get_weights()) +
                                          (1 - self.alpha) * np.array(target_actor.get_weights())))
                target_critic.set_weights((self.alpha * np.array(critic.get_weights()) +
                                           (1 - self.alpha) * np.array(target_critic.get_weights())))

                # For logging data
                if done or current_step > max_step:
                    visualize(episode_reward, episode_count, slide_window, "DDPG.png")
                    break

                state = next_state
                current_step += 1

if __name__ == '__main__':
    agent = DDPG()
    agent.train()
