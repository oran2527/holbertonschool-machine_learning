# 0x01. Deep Q-learning

## Holberton Cali

## 10 august 2021

## Orlando Gomez Lopez

## Machine Learning

## Cohort 10

0x01. Deep Q-learning
 Specializations > Machine Learning > Reinforcement Learning
 By Alexa Orrico, Software Engineer at Holberton School
 Ongoing project - started 08-04-2021, must end by 08-13-2021 (in 3 days) - you're done with 0% of tasks.
 Manual QA review must be done (request it when you are done with the project)


Resources
Read or watch:

Deep Q-Learning - Combining Neural Networks and Reinforcement Learning
Replay Memory Explained - Experience for Deep Q-Network Training
Training a Deep Q-Network - Reinforcement Learning
Training a Deep Q-Network with Fixed Q-targets - Reinforcement Learning
References:

Setting up anaconda for keras-rl
keras-rl
rl.policy
rl.memory
rl.agents.dqn
Playing Atari with Deep Reinforcement Learning
Learning Objectives
What is Deep Q-learning?
What is the policy network?
What is replay memory?
What is the target network?
Why must we utilize two separate networks during training?
What is keras-rl? How do you use it?
Requirements
General
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
Your files will be executed with numpy (version 1.15), gym (version 0.17.2), keras (version 2.2.5), and keras-rl (version 0.4.2)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.4)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
All your files must be executable
Your code should use the minimum number of operations
Installing Keras-RL
pip install --user keras-rl
Dependencies (that should already be installed)
pip install --user keras==2.2.4
pip install --user Pillow
pip install --user h5py

## Tasks

## 0. Breakout

mandatory
Write a python script train.py that utilizes keras, keras-rl, and gym to train an agent that can play Atari’s Breakout:

Your script should utilize keras-rl‘s DQNAgent, SequentialMemory, and EpsGreedyQPolicy
Your script should save the final policy network as policy.h5
Write a python script play.py that can display a game played by the agent trained by train.py:

Your script should load the policy network saved in policy.h5
Your agent should use the GreedyQPolicy
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: reinforcement_learning/0x01-deep_q_learning
File: train.py, play.py
