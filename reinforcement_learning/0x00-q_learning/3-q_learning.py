#!/usr/bin/env python3
""" Q-learning """
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """ Q-learning """
    
    rewards = []

    
    for episode in range(episodes):
        
        state = env.reset()
        total_rewards = 0

        for step in range(max_steps):
            
            exp_exp_tradeoff = np.random.uniform(0, 1)

            
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(Q[state, :])

            
            else:
                action = env.action_space.sample()

            
            new_state, reward, done, info = env.step(action)

            
            Q[state, action] = Q[state, action] + alpha *\
                (reward + gamma *
                 np.max(Q[new_state, :]) -
                 Q[state, action])

            total_rewards += reward

            
            state = new_state

            
            if done:
                break

        
        epsilon = min_epsilon + (epsilon - min_epsilon) *\
            np.exp(-epsilon_decay * episode)
        rewards.append(total_rewards)
    env.close()
    return Q, rewards
