# 0x03. Policy Gradients

## Holberton Cali

## 25 august 2021

## Orlando Gomez Lopez

## Machine Learning

## Cohort 10

0x03. Policy Gradients
 By Alexa Orrico, Software Engineer at Holberton School
 Ongoing project - started 08-23-2021, must end by 08-27-2021 (in 2 days) - you're done with 0% of tasks.
 Manual QA review must be done (request it when you are done with the project)


In this project, you will implement your own Policy Gradient in your loop of reinforcement learning (by using the Monte-Carlo policy gradient algorithm - also called REINFORCE).

Resources
Read or watch:

How Policy Gradient Reinforcement Learning Works
Policy Gradients in a Nutshell
RL Course by David Silver - Lecture 7: Policy Gradient Methods
Reinforcement Learning 6: Policy Gradients and Actor Critics
Policy Gradient Algorithms
Learning Objectives
What is Policy?
How to calculate a Policy Gradient?
What and how to use a Monte-Carlo policy gradient?
Requirements
General
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
Your files will be executed with numpy (version 1.15), and gym (version 0.7)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.4)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
All your files must be executable
Your code should use the minimum number of operations

## Tasks

## 0. Simple Policy function

mandatory
Write a function that computes to policy with a weight of a matrix.

Prototype: def policy(matrix, weight):
$ cat 0-main.py
#!/usr/bin/env python3
"""
Main file
"""
import numpy as np
from policy_gradient import policy


weight = np.ndarray((4, 2), buffer=np.array([
    [4.17022005e-01, 7.20324493e-01], 
    [1.14374817e-04, 3.02332573e-01], 
    [1.46755891e-01, 9.23385948e-02], 
    [1.86260211e-01, 3.45560727e-01]
    ]))
state = np.ndarray((1, 4), buffer=np.array([
    [-0.04428214,  0.01636746,  0.01196594, -0.03095031]
    ]))

res = policy(state, weight)
print(res)

$
$ ./0-main.py
[[0.50351642 0.49648358]]
$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: reinforcement_learning/0x03-policy_gradients
File: policy_gradient.py
 
## 1. Compute the Monte-Carlo policy gradient

mandatory
By using the previous function created policy, write a function that computes the Monte-Carlo policy gradient based on a state and a weight matrix.

Prototype: def policy_gradient(state, weight):
state: matrix representing the current observation of the environment
weight: matrix of random weight
Return: the action and the gradient (in this order)
$ cat 1-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym
import numpy as np
from policy_gradient import policy_gradient

env = gym.make('CartPole-v1')
np.random.seed(1)

weight = np.random.rand(4, 2)
state = env.reset()[None,:]
print(weight)
print(state)

action, grad = policy_gradient(state, weight)
print(action)
print(grad)

env.close()

$ 
$ ./1-main.py
[[4.17022005e-01 7.20324493e-01]
 [1.14374817e-04 3.02332573e-01]
 [1.46755891e-01 9.23385948e-02]
 [1.86260211e-01 3.45560727e-01]]
[[ 0.04228739 -0.04522399  0.01190918 -0.03496226]]
0
[[ 0.02106907 -0.02106907]
 [-0.02253219  0.02253219]
 [ 0.00593357 -0.00593357]
 [-0.01741943  0.01741943]]
$ 
*Results can be different since weight is randomized *

Repo:

GitHub repository: holbertonschool-machine_learning
Directory: reinforcement_learning/0x03-policy_gradients
File: policy_gradient.py
 
## 2. Implement the training

mandatory
By using the previous function created policy_gradient, write a function that implements a full training.

Prototype: def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
env: initial environment
nb_episodes: number of episodes used for training
alpha: the learning rate
gamma: the discount factor
Return: all values of the score (sum of all rewards during one episode loop)
Since the training is quite long, please print the current episode number and the score after each loop. To display these information on the same line, you can use end="\r", flush=False of the print function.

With the following main file, you should have this result plotted:



$ cat 2-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym
import matplotlib.pyplot as plt
import numpy as np

from train import train

env = gym.make('CartPole-v1')

scores = train(env, 10000)

plt.plot(np.arange(len(scores)), scores)
plt.show()
env.close()

$ 
$ ./2-main.py
Results can be different we have multiple randomization

Also, we highly encourage you to play with alpha and gamma to change the trend of the plot

Repo:

GitHub repository: holbertonschool-machine_learning
Directory: reinforcement_learning/0x03-policy_gradients
File: train.py
 
## 3. Animate iteration

mandatory
Update the prototype of the train function by adding a last optional parameter show_result (default: False).

When this parameter is True, render the environment every 1000 episodes computed.

$ cat 3-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym

from train import train

env = gym.make('CartPole-v1')

scores = train(env, 10000, 0.000045, 0.98, True)

env.close()

$ 
$ ./3-main.py
Results can be different we have multiple randomization

Result after few episodes:



Result after more episodes:



Result after 10000 episodes:



Repo:

GitHub repository: holbertonschool-machine_learning
Directory: reinforcement_learning/0x03-policy_gradients
File: train.py
