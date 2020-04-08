# Dueling Double DQN with Experience Replay using Proportional method.
# Noisy is added at the end of advantage node to encourage exploration.

import gym
import numpy as np
import random
from keras.layers import Dense, Input, Activation, Lambda
from keras import Model, activations, initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras.optimizers import Adam
import keras.backend as K
from itertools import count
from util import visualization
import tensorflow as tf


# Noisy netowrk for exploration.
# Code from https://github.com/LuEE-C/Noisy-A3C-Keras/blob/master/NoisyDense.py
# Fixed by https://github.com/keiohta/tf2rl/blob/master/tf2rl/networks/noisy_dense.py
class NoisyDense(Layer):

    def __init__(self, units,
                 sigma_init=0.02,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.sigma_init = sigma_init
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        self.kernel_shape = tf.constant((self.input_dim, self.units))
        self.bias_shape = tf.constant((self.units,))

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.sigma_kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=initializers.Constant(value=self.sigma_init),
                                      name='sigma_kernel'
                                      )


        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.sigma_bias = self.add_weight(shape=(self.units,),
                                        initializer=initializers.Constant(value=self.sigma_init),
                                        name='sigma_bias')
        else:
            self.bias = None
            self.epsilon_bias = None

        self.epsilon_kernel = K.zeros(shape=(self.input_dim, self.units))
        self.epsilon_bias = K.zeros(shape=(self.units,))

        self.sample_noise()
        super(NoisyDense, self).build(input_shape)


    def call(self, X):
        perturbation = self.sigma_kernel * K.random_uniform(shape=self.kernel_shape)
        perturbed_kernel = self.kernel + perturbation
        output = K.dot(X, perturbed_kernel)
        if self.use_bias:
            bias_perturbation = self.sigma_bias * K.random_uniform(shape=self.bias_shape)
            perturbed_bias = self.bias + bias_perturbation
            output = K.bias_add(output, perturbed_bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def sample_noise(self):
        K.set_value(self.epsilon_kernel, np.random.normal(0, 1, (self.input_dim, self.units)))
        K.set_value(self.epsilon_bias, np.random.normal(0, 1, (self.units,)))

    def remove_noise(self):
        K.set_value(self.epsilon_kernel, np.zeros(shape=(self.input_dim, self.units)))
        K.set_value(self.epsilon_bias, np.zeros(shape=self.units,))


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


class Noisy_DDDQN_PER_Prop():
    def __init__(self):
        self.gamma = 0.95   # reward discount
        self.learning_rate = 0.001
        self.memory_size = 10000
        self.epsilon = 0.8  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.memory = SumTree(self.memory_size)
        self.batch_size = 32
        self.rewards = []
        self.update_frequency = 50
        self.counter = count()  # In case multiple memories with same priority
        self.replay_frequency = 10
        self.alpha = 0.6    # Used during priority calculation
        self.beta = 0.4  # Used during importance sampling
        self.beta_increase = 0.001
        self.beta_max = 1
        self.constant = 1e-10   # In case priority is 0

    def build_network(self):
        input = Input(shape=(self.num_states,))
        layer = Dense(24, activation='relu')(input)
        layer_v = Dense(1)(layer)
        layer_a = NoisyDense(self.num_actions)(layer)
        v = Activation('linear')(layer_v)
        a = Activation('linear')(layer_a)
        output = Lambda(lambda x: x[0] + x[1] - K.mean(x[1], axis=1, keepdims=True), output_shape=(self.num_actions,))([v, a])  # Q = V + A
        model = Model(input, output)
        adam = Adam(lr=self.learning_rate)

        def huber_loss(y_true, y_pred):
            return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1)

        model.compile(loss=huber_loss, optimizer=adam)  # Not sure if using huber loss is good here

        return model

    # Save <s, a ,r, s'> of each step
    def store_memory(self, priority, state, action, reward, next_state, done):
        self.memory.add(priority, [state, action, reward, next_state, done])

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

                # Network predict
                q_values = q_network.predict(np.reshape(state, (1, self.num_states))).ravel()
                action = np.argmax(q_values)

                # Perform action
                next_state, reward, done, info = env.step(action)

                # Calculate priority
                next_q_values = target_q_network.predict(np.reshape(next_state, (1, self.num_states))).ravel()
                next_action = np.argmax(q_network.predict(np.reshape(next_state, (1, self.num_states))).ravel())
                td_error = reward + self.gamma * (1 - done) * next_q_values[next_action] - q_values[action] # Note that the td_error is not ^2 like in DQN
                priority = (abs(td_error) + self.constant) ** self.alpha

                # Store transition
                episode_reward += reward
                self.store_memory(priority, state, action, reward, next_state, done)

                if current_step % self.replay_frequency is 0:
                    # Sample minibatch from memory based on their priority
                    minibatch = []
                    ISWeights = []
                    min_prob = np.min(self.memory.tree[-self.memory.capacity] / self.memory.total())
                    T = self.memory.total() // self.batch_size
                    for i in range(self.batch_size):
                        a, b = T * i, T * (i + 1)
                        s = random.uniform(a, b)
                        idx, priority, data = self.memory.get(s)
                        probability = priority/self.memory.total()
                        ISWeights.append(np.power(probability/min_prob, -self.beta))
                        minibatch.append((*data, idx))
                    self.beta = np.min([self.beta_max, self.beta + self.beta_increase])

                    # Transform the minibatch for processing
                    minibatch = list(zip(*minibatch))

                    # Calculate all td_targets for current minibatch
                    states, actions, rewards, next_states, dones, indices = minibatch
                    batch_q_values = q_network.predict_on_batch(np.array(states))
                    batch_next_q_values = target_q_network.predict_on_batch(np.array(next_states))
                    next_actions = np.argmax(q_network.predict_on_batch(np.array(next_states)), axis=1)
                    td_targets = batch_q_values.copy()
                    for i in range(self.batch_size):
                        td_targets[i][actions[i]] = rewards[i] + self.gamma * (1 - dones[i]) * batch_next_q_values[i][next_actions[i]]
                        # Need to recalculate priorities for transitions in minibatch
                        priority = (abs(td_targets[i][actions[i]] - batch_q_values[i][actions[i]]) + self.constant) ** self.alpha
                        self.memory.update(indices[i], priority)

                    # Train network
                    q_network.train_on_batch(np.array(states), np.array(td_targets), np.array(ISWeights))

                    # Hard copy q_network to target_q_network
                    if done or current_step % self.update_frequency is 0:
                        target_q_network.set_weights(q_network.get_weights())

                # For logging data
                if done or current_step > max_step:
                    visualization(episode_reward, episode_count, slide_window, "Noisy_DDDQN_PER_Prop.png")
                    break

                state = next_state
                current_step += 1

if __name__ == '__main__':
    agent = Noisy_DDDQN_PER_Prop()
    agent.train()