import random

import numpy as np
import tensorflow as tf

from keras import Sequential
from collections import deque
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from main_game_11 import Game
from player_movement import moves


class DQN:

    """ Deep Q Network """

    def __init__(self, env, params):

        self.action_space = env.action_space
        self.state_space = env.state_space
        self.epsilon = params['epsilon'] 
        self.gamma = params['gamma'] 
        self.batch_size = params['batch_size'] 
        self.epsilon_min = params['epsilon_min'] 
        self.epsilon_decay = params['epsilon_decay'] 
        self.learning_rate = params['learning_rate']
        #self.layer_sizes = params['layer_sizes']
        self.memory = deque(maxlen=2500)
        self.model = self.build_model(params['act_function'])
        
        self.pretrained_act = False


    def build_model(self, act_function):
        
        init = tf.keras.initializers.HeUniform()
        model = Sequential()
        model.add(Dense(16, input_shape=(self.state_space,), activation=act_function, kernel_initializer=init))
        model.add(Dense(32, activation=act_function, kernel_initializer=init))
        model.add(Dense(64, activation=act_function, kernel_initializer=init))
        model.add(Dense(self.action_space, activation='linear', kernel_initializer=init))
        opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.5, decay=1e-6, clipnorm=2)
        #model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
        
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    
    def act2(self, index):
        return moves[index]


    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode, env, pic_episode_list):
    sum_of_rewards = []
    agent = DQN(env, params)
    for e in range(episode):
        state = env.reset()
        if e in pic_episode_list:
            env.take_pics = True
        print('Will it save frames = {}'.format(env.take_pics))
        state = np.reshape(state, (1, env.state_space))
        score = 0
        max_steps = 10000
        if e in episode_pretrain_list:
            agent.pretrained_act = True
        print('Is this EP pretrained = {}'.format(agent.pretrained_act))
        for i in range(max_steps):
            if agent.pretrained_act: 
                action = agent.act2(i)
            elif agent.pretrained_act == False:
                action = agent.act(state)
            # print(action)
            prev_state = state
            next_state, reward, dead, _ = env.step(action, e+1)
            score += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            agent.remember(state, action, reward, next_state, dead)
            state = next_state
            if params['batch_size'] > 1:
                agent.replay()
            if dead:
                adj_score = round((score / env.boat.lifespan), 2)
                print(f'final state before dying: {str(prev_state)}')
                print(f'episode: {e+1}/{episode}, score: {score}, ADJscore: {adj_score}')
                print(f'Points: {env.boat.score}')
                break
        sum_of_rewards.append(score)
        agent.pretrained_act = False
    env.close()
    return sum_of_rewards


if __name__ == '__main__':

    params = dict()
    params['name'] = None
    params['epsilon'] = 1
    params['gamma'] = .95
    params['batch_size'] = 64
    params['epsilon_min'] = .01
    params['epsilon_decay'] = .995
    params['learning_rate'] = 0.00025
    #params['act_function'] = 'relu'
    params['act_function'] = 'tanh'
    
    pic_episode_list = [0, 9,   19,   29,   39,   49,   59,   69,   79,   89,   99,
                         109,  119,  129,  139,  149,  159,  169,  179,  189,  199,
                         249,  299,  349,  399,  449,  499,  549,  599,  649,  699,
                         749,  799,  849,  899,  949,  999, 1049, 1099, 1149, 1199,
                        1249, 1299, 1349, 1399, 1449, 1499, 1549, 1599, 1649, 1699,
                        1749, 1799, 1849, 1899, 1949, 1999, 
                        ]

    results = dict()
    ep = 2000#50
    
    episode_pretrain_list = []#1, 9, 24, 49, 98, 186, 342, 624]
    
    env = Game('ai')
    sum_of_rewards = train_dqn(ep, env, pic_episode_list)
    results[params['name']] = sum_of_rewards
    
    #plot_result(results, direct=True, k=20)