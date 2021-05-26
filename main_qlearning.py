import os
import random
import time
import numpy as np

import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque


from snake import Snake
from plot_script import plot_result

from utils.config import *



class DeepQLearning(object):
    '''
    Deep Q Network
    '''

    def __init__(self, env):
        # Game params
        self.action_space = env.action_space
        self.state_space  = env.state_space
        
        # Params for Q-learning
        self.epsilon       = EPSILON
        self.discount_rate = DISCOUNT_RATE
        self.batch_size    = BATCH_SIZE
        self.epsilon_min   = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        
        # Memory Buffer
        self.replay_buffer = deque(maxlen=2500)

        # Neural Network model
        self.layer_units = LAYER_UNITS
        self.network     = self.network()


    def network(self):
        network = Sequential()
        
        for i in range(len(self.layer_units)):
            if i == 0:
                # Input
                network.add(Dense(self.layer_units[i], input_shape=(self.state_space,), activation='relu'))
            else:
                # Hidden layers
                network.add(Dense(self.layer_units[i], activation='relu'))
        
        # Output layer
        network.add(Dense(self.action_space, activation='softmax'))
        
        network.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return network


    def remember(self, state, action, reward, next_state, done):
        '''
        Instead of training the DQN based only on the latest experiences, we will store all experiences in a replay buffer (or replay memory), and we will sample a random training batch from it at each training iteration. This helps reduce the correlations between the experiences in a training batch, which tremendously helps training. For this, we will just use a deque list:
        We will also need a replay memory. It will contain the agent's experiences, in the form of tuples: (obs, action, reward, next_obs, done). We can use the deque class for that:
        '''
        self.replay_buffer.append((state, action, reward, next_state, done))


    def epsilon_greedy_policy(self, state):        
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)
        
        Q_values = self.network.predict(state[np.newaxis])
        return np.argmax(Q_values[0])


    def batch_sample(self):
        batch = random.sample(self.replay_buffer, self.batch_size)

        states, actions, rewards, next_states, dones =\
               [np.array([experience[field_index] for experience in batch]) for field_index in range(5)]

        return states, actions, rewards, next_states, dones




    def batch_training(self):

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.batch_sample()
        states      = np.squeeze(states)
        next_states = np.squeeze(next_states)

        next_Q_values = self.network.predict_on_batch(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)

        target_Q_values = (rewards + (1 - dones) * self.discount_rate * max_next_Q_values)

        target_Q_values_full = self.network.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        target_Q_values_full[[ind], [actions]] = target_Q_values

        self.network.fit(states, target_Q_values_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



def play_one_step(agent, env, state):
  action = agent.epsilon_greedy_policy(state)
  next_state, reward, done, info = env.step(action)
  agent.remember(state, action, reward, next_state, done)
  return next_state, reward, done, info



def train(env, episode = 500):
    sum_of_rewards = []
    agent = DeepQLearning(env)
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, (1, env.state_space))
        score = 0
        max_steps = 10000

        for i in range(max_steps):
            prev_state = state
            next_state, reward, done, _ = play_one_step(agent, env, state)
            score += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            state = next_state
            
            if episode > 50:
                agent.batch_training()
            
            if done:
                print(f'Final state before dying: {str(prev_state)}')
                print(f'Episode: {episode+1}/{episodes}, score: {score}')
                break
        sum_of_rewards.append(score)
    return sum_of_rewards






def save_fig(img_path, fig_id, tight_layout=True, fig_extension='png', resolution=300):
  path = os.path.join(img_path, fig_id + '.' + fig_extension)
  print('Saving figure', fig_id)
  if tight_layout:
    plt.tight_layout()
  plt.savefig(path, format=fig_extension, dpi=resolution)    





if __name__ == '__main__':    

    # Training
    episodes = 120
    env = Snake()
    sum_of_rewards = train(env, episode = episodes)
    
    
    # Where to save the figure
    img_path = 'images/'
    os.makedirs(img_path, exist_ok=True)

    # Figure
    plt.figure( figsize=(8, 4) )
    plt.plot(sum_of_rewards)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Sum of rewards', fontsize=14)

    save_fig(img_path, 'sum_of_rewards', tight_layout=True, fig_extension='png', resolution=300)

    plt.show()