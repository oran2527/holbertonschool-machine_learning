#!/usr/bin/env python3
""" Sarsa Lambtha """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ Sarsa Lambtha """
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(1)
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """ Sarsa Lambtha """
    n = env.observation_space.n
    eps = epsilon
    Et = np.zeros((Q.shape))
    for i in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        for _ in range(max_steps):
            Et *= lambtha * gamma
            Et[state, action] += 1
            new_state, reward, done, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon)
            if env.desc.reshape(n)[new_state] == b'G':
                reward = 1
            if env.desc.reshape(n)[new_state] == b'H':
                reward = -1
            deltat = (reward + gamma * Q[new_state, new_action]
                      - Q[state, action])
            Q[state, action] = (Q[state, action] + alpha
                                * deltat * Et[state, action])
            if done:
                break
            state = new_state
            action = new_action
        epsilon = (min_epsilon + (eps - min_epsilon)
                   * np.exp(-epsilon_decay * i))
    return Q
