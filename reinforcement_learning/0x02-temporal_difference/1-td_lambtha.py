#!/usr/bin/env python3
import numpy as np
""" TD(λ) algorithm """

def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """ TD(λ) algorithm """
    n = env.observation_space.n
    Et = [0 for i in range(n)]
    for _ in range(episodes):
        state = env.reset()
        for step in range(max_steps):
            Et = list(np.array(Et) * lambtha * gamma)
            Et[state] += 1.0
            action = policy(state)
            new_state, reward, done, info = env.step(action)
            if env.desc.reshape(n)[new_state] == b'G':
                reward = 1
            if env.desc.reshape(n)[new_state] == b'H':
                reward = -1            
            deltat = reward + gamma * V[new_state] - V[state]            
            V[state] = V[state] + alpha * deltat * Et[state]
            if done:
                break
            state = new_state
    return V
