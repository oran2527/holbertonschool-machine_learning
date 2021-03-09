# 0x06. Keras

## Holberton Cali

## 9 march 2021

## Orlando Gomez Lopez

## Machine Learning

## Cohort 10

0x06. Keras

 Specializations - Machine Learning ― Supervised Learning
 By Alexa Orrico, Software Engineer at Holberton School
 Ongoing project - started 03-08-2021, must end by 03-10-2021 (in about 18 hours) - you're done with 0% of tasks.
 Checker was released at 03-09-2021 12:00 AM
 QA review fully automated.


Resources
Read or watch:

Keras Explained (starting at 3:48)
Keras (up to Eager execution, excluded, skipping Train from tf.data datasets, Model subclassing, and Custom layers)
Hierarchical Data Format
References:

tf.keras
tf.keras.models
tf.keras.activations
tf.keras.callbacks
tf.keras.initializers
tf.keras.layers
tf.keras.losses
tf.keras.metrics
tf.keras.optimizers
tf.keras.regularizers
tf.keras.utils
Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
What is Keras?
What is a model?
How to instantiate a model (2 ways)
How to build a layer
How to add regularization to a layer
How to add dropout to a layer
How to add batch normalization
How to compile a model
How to optimize a model
How to fit a model
How to use validation data
How to perform early stopping
How to measure accuracy
How to evaluate a model
How to make a prediction with a model
How to access the weights/outputs of a model
What is HDF5?
How to save and load a model’s weights, a model’s configuration, and the entire model
Requirements
General
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
Your files will be executed with numpy (version 1.15) and tensorflow (version 1.12)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.4)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
Unless otherwise noted, you are not allowed to import any module except import tensorflow.keras as K
All your files must be executable
The length of your files will be tested using wc

## Tasks

## 0. Sequential

mandatory
Write a function def build_model(nx, layers, activations, lambtha, keep_prob): that builds a neural network with the Keras library:

nx is the number of input features to the network
layers is a list containing the number of nodes in each layer of the network
activations is a list containing the activation functions used for each layer of the network
lambtha is the L2 regularization parameter
keep_prob is the probability that a node will be kept for dropout
You are not allowed to use the Input class
Returns: the keras model
ubuntu@alexa-ml:~/0x06-keras$ cat 0-main.py 
#!/usr/bin/env python3

build_model = __import__('0-sequential').build_model

if __name__ == '__main__':
    network = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    network.summary()
    print(network.losses)
ubuntu@alexa-ml:~/0x06-keras$ ./0-main.py 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 256)               200960    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 269,322
Trainable params: 269,322
Non-trainable params: 0
_________________________________________________________________
[<tf.Tensor 'dense/kernel/Regularizer/add:0' shape=() dtype=float32>, <tf.Tensor 'dense_1/kernel/Regularizer/add:0' shape=() dtype=float32>, <tf.Tensor 'dense_2/kernel/Regularizer/add:0' shape=() dtype=float32>]
ubuntu@alexa-ml:~/0x06-keras$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x06-keras
File: 0-sequential.py
   
## 1. Input

mandatory
Write a function def build_model(nx, layers, activations, lambtha, keep_prob): that builds a neural network with the Keras library:

nx is the number of input features to the network
layers is a list containing the number of nodes in each layer of the network
activations is a list containing the activation functions used for each layer of the network
lambtha is the L2 regularization parameter
keep_prob is the probability that a node will be kept for dropout
You are not allowed to use the Sequential class
Returns: the keras model
ubuntu@alexa-ml:~/0x06-keras$ cat 1-main.py 
#!/usr/bin/env python3

build_model = __import__('1-input').build_model

if __name__ == '__main__':
    network = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    network.summary()
    print(network.losses)
ubuntu@alexa-ml:~/0x06-keras$ ./1-main.py 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               200960    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 269,322
Trainable params: 269,322
Non-trainable params: 0
_________________________________________________________________
[<tf.Tensor 'dense/kernel/Regularizer/add:0' shape=() dtype=float32>, <tf.Tensor 'dense_2/kernel/Regularizer/add:0' shape=() dtype=float32>, <tf.Tensor 'dense_1/kernel/Regularizer/add:0' shape=() dtype=float32>]
ubuntu@alexa-ml:~/0x06-keras$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x06-keras
File: 1-input.py
   
## 2. Optimize

mandatory
Write a function def optimize_model(network, alpha, beta1, beta2): that sets up Adam optimization for a keras model with categorical crossentropy loss and accuracy metrics:

network is the model to optimize
alpha is the learning rate
beta1 is the first Adam optimization parameter
beta2 is the second Adam optimization parameter
Returns: None
ubuntu@alexa-ml:~/0x06-keras$ cat 2-main.py 
#!/usr/bin/env python3

import tensorflow as tf

build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model

if __name__ == '__main__':
    model = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    optimize_model(model, 0.01, 0.99, 0.9)
    print(model.loss)
    print(model.metrics)
    opt = model.optimizer
    print(opt.__class__)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run((opt.lr, opt.beta_1, opt.beta_2))) 

ubuntu@alexa-ml:~/0x06-keras$ ./2-main.py 
categorical_crossentropy
['accuracy']
<class 'tensorflow.python.keras.optimizers.Adam'>
(0.01, 0.99, 0.9)
ubuntu@alexa-ml:~/0x06-keras$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x06-keras
File: 2-optimize.py
   
## 3. One Hot

mandatory
Write a function def one_hot(labels, classes=None): that converts a label vector into a one-hot matrix:

The last dimension of the one-hot matrix must be the number of classes
Returns: the one-hot matrix
ubuntu@alexa-ml:~/0x06-keras$ cat 3-main.py 
#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot

if __name__ == '__main__':
    labels = np.load('../data/MNIST.npz')['Y_train'][:10]
    print(labels)
    print(one_hot(labels))   
ubuntu@alexa-ml:~/0x06-keras$ ./3-main.py 
[5 0 4 1 9 2 1 3 1 4]
[[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
ubuntu@alexa-ml:~/0x06-keras$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x06-keras
File: 3-one_hot.py
   
## 4. Train

mandatory
Write a function def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False): that trains a model using mini-batch gradient descent:

network is the model to train
data is a numpy.ndarray of shape (m, nx) containing the input data
labels is a one-hot numpy.ndarray of shape (m, classes) containing the labels of data
batch_size is the size of the batch used for mini-batch gradient descent
epochs is the number of passes through data for mini-batch gradient descent
verbose is a boolean that determines if output should be printed during training
shuffle is a boolean that determines whether to shuffle the batches every epoch. Normally, it is a good idea to shuffle, but for reproducibility, we have chosen to set the default to False.
Returns: the History object generated after training the model
ubuntu@alexa-ml:~/0x06-keras$ cat 4-main.py
#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('4-train').train_model


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)

    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)
    batch_size = 64
    epochs = 5
    train_model(network, X_train, Y_train_oh, batch_size, epochs)
ubuntu@alexa-ml:~/0x06-keras$ ./4-main.py
Epoch 1/5
50000/50000 [==============================] - 7s 140us/step - loss: 0.3508 - acc: 0.9202
Epoch 2/5
50000/50000 [==============================] - 5s 106us/step - loss: 0.1964 - acc: 0.9660
Epoch 3/5
50000/50000 [==============================] - 5s 103us/step - loss: 0.1587 - acc: 0.9760
Epoch 4/5
50000/50000 [==============================] - 7s 135us/step - loss: 0.1374 - acc: 0.9810
Epoch 5/5
50000/50000 [==============================] - 6s 117us/step - loss: 0.1242 - acc: 0.9837
ubuntu@alexa-ml:~/0x06-keras$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x06-keras
File: 4-train.py
   
## 5. Validate

mandatory
Based on 4-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False): to also analyze validaiton data:

validation_data is the data to validate the model with, if not None
ubuntu@alexa-ml:~/0x06-keras$ cat 5-main.py 
#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('5-train').train_model

if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)
    X_valid = datasets['X_valid']
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = datasets['Y_valid']
    Y_valid_oh = one_hot(Y_valid)

    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)
    batch_size = 64
    epochs = 5
    train_model(network, X_train, Y_train_oh, batch_size, epochs, validation_data=(X_valid, Y_valid_oh))

ubuntu@alexa-ml:~/0x06-keras$ ./5-main.py 
Train on 50000 samples, validate on 10000 samples
Epoch 1/5
50000/50000 [==============================] - 7s 145us/step - loss: 0.3508 - acc: 0.9202 - val_loss: 0.2174 - val_acc: 0.9602
Epoch 2/5
50000/50000 [==============================] - 7s 135us/step - loss: 0.1964 - acc: 0.9660 - val_loss: 0.1772 - val_acc: 0.9702
Epoch 3/5
50000/50000 [==============================] - 7s 131us/step - loss: 0.1587 - acc: 0.9760 - val_loss: 0.1626 - val_acc: 0.9740
Epoch 4/5
50000/50000 [==============================] - 6s 129us/step - loss: 0.1374 - acc: 0.9810 - val_loss: 0.1783 - val_acc: 0.9703
Epoch 5/5
50000/50000 [==============================] - 7s 137us/step - loss: 0.1242 - acc: 0.9837 - val_loss: 0.1547 - val_acc: 0.9757
ubuntu@alexa-ml:~/0x06-keras$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x06-keras
File: 5-train.py
   
## 6. Early Stopping

mandatory
Based on 5-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False): to also train the model using early stopping:

early_stopping is a boolean that indicates whether early stopping should be used
early stopping should only be performed if validation_data exists
early stopping should be based on validation loss
patience is the patience used for early stopping
ubuntu@alexa-ml:~/0x06-keras$ cat 6-main.py 
#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('6-train').train_model


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)
    X_valid = datasets['X_valid']
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = datasets['Y_valid']
    Y_valid_oh = one_hot(Y_valid)

    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)
    batch_size = 64
    epochs = 30
    train_model(network, X_train, Y_train_oh, batch_size, epochs,
                validation_data=(X_valid, Y_valid_oh), early_stopping=True,
                patience=3)

ubuntu@alexa-ml:~/0x06-keras$ ./6-main.py 
Train on 50000 samples, validate on 10000 samples
Epoch 1/30
50000/50000 [==============================] - 6s 126us/step - loss: 0.3508 - acc: 0.9202 - val_loss: 0.2174 - val_acc: 0.9602
Epoch 2/30
50000/50000 [==============================] - 6s 112us/step - loss: 0.1964 - acc: 0.9660 - val_loss: 0.1772 - val_acc: 0.9702
Epoch 3/30
50000/50000 [==============================] - 5s 108us/step - loss: 0.1587 - acc: 0.9760 - val_loss: 0.1626 - val_acc: 0.9740
Epoch 4/30
50000/50000 [==============================] - 7s 134us/step - loss: 0.1374 - acc: 0.9810 - val_loss: 0.1783 - val_acc: 0.9703
Epoch 5/30
50000/50000 [==============================] - 7s 137us/step - loss: 0.1242 - acc: 0.9837 - val_loss: 0.1547 - val_acc: 0.9757
Epoch 6/30
50000/50000 [==============================] - 7s 143us/step - loss: 0.1185 - acc: 0.9848 - val_loss: 0.1547 - val_acc: 0.9754
Epoch 7/30
50000/50000 [==============================] - 8s 155us/step - loss: 0.1114 - acc: 0.9863 - val_loss: 0.1477 - val_acc: 0.9779
Epoch 8/30
50000/50000 [==============================] - 8s 163us/step - loss: 0.1046 - acc: 0.9879 - val_loss: 0.1469 - val_acc: 0.9777
Epoch 9/30
50000/50000 [==============================] - 7s 131us/step - loss: 0.1016 - acc: 0.9885 - val_loss: 0.1437 - val_acc: 0.9803
Epoch 10/30
50000/50000 [==============================] - 6s 115us/step - loss: 0.0979 - acc: 0.9899 - val_loss: 0.1464 - val_acc: 0.9759
Epoch 11/30
50000/50000 [==============================] - 5s 104us/step - loss: 0.0968 - acc: 0.9892 - val_loss: 0.1448 - val_acc: 0.9771
Epoch 12/30
50000/50000 [==============================] - 6s 119us/step - loss: 0.0977 - acc: 0.9887 - val_loss: 0.1378 - val_acc: 0.9804
Epoch 13/30
50000/50000 [==============================] - 6s 125us/step - loss: 0.0915 - acc: 0.9911 - val_loss: 0.1434 - val_acc: 0.9789
Epoch 14/30
50000/50000 [==============================] - 6s 113us/step - loss: 0.0916 - acc: 0.9903 - val_loss: 0.1373 - val_acc: 0.9799
Epoch 15/30
50000/50000 [==============================] - 7s 137us/step - loss: 0.0910 - acc: 0.9907 - val_loss: 0.1369 - val_acc: 0.9791
Epoch 16/30
50000/50000 [==============================] - 6s 115us/step - loss: 0.0893 - acc: 0.9903 - val_loss: 0.1371 - val_acc: 0.9790
Epoch 17/30
50000/50000 [==============================] - 5s 104us/step - loss: 0.0869 - acc: 0.9910 - val_loss: 0.1423 - val_acc: 0.9789
Epoch 18/30
50000/50000 [==============================] - 8s 160us/step - loss: 0.0882 - acc: 0.9911 - val_loss: 0.1375 - val_acc: 0.9798
ubuntu@alexa-ml:~/0x06-keras$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x06-keras
File: 6-train.py
   
## 7. Learning Rate Decay

mandatory
Based on 6-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False): to also train the model with learning rate decay:

learning_rate_decay is a boolean that indicates whether learning rate decay should be used
learning rate decay should only be performed if validation_data exists
the decay should be performed using inverse time decay
the learning rate should decay in a stepwise fashion after each epoch
each time the learning rate updates, Keras should print a message
alpha is the initial learning rate
decay_rate is the decay rate
ubuntu@alexa-ml:~/0x06-keras$ cat 7-main.py 
#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('7-train').train_model 

if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)
    X_valid = datasets['X_valid']
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = datasets['Y_valid']
    Y_valid_oh = one_hot(Y_valid)

    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)
    batch_size = 64
    epochs = 1000
    train_model(network, X_train, Y_train_oh, batch_size, epochs,
                validation_data=(X_valid, Y_valid_oh), early_stopping=True,
                patience=3, learning_rate_decay=True, alpha=alpha)

ubuntu@alexa-ml:~/0x06-keras$ ./7-main.py
Train on 50000 samples, validate on 10000 samples

Epoch 00001: LearningRateScheduler reducing learning rate to 0.001.
Epoch 1/1000
50000/50000 [==============================] - 7s 149us/step - loss: 0.3508 - acc: 0.9202 - val_loss: 0.2174 - val_acc: 0.9602

Epoch 00002: LearningRateScheduler reducing learning rate to 0.0005.
Epoch 2/1000
50000/50000 [==============================] - 6s 112us/step - loss: 0.1823 - acc: 0.9705 - val_loss: 0.1691 - val_acc: 0.9743

Epoch 00003: LearningRateScheduler reducing learning rate to 0.0003333333333333333.
Epoch 3/1000
50000/50000 [==============================] - 8s 151us/step - loss: 0.1481 - acc: 0.9795 - val_loss: 0.1563 - val_acc: 0.9769

Epoch 00004: LearningRateScheduler reducing learning rate to 0.00025.
Epoch 4/1000
50000/50000 [==============================] - 6s 129us/step - loss: 0.1287 - acc: 0.9849 - val_loss: 0.1499 - val_acc: 0.9770

...

Epoch 00065: LearningRateScheduler reducing learning rate to 1.5384615384615384e-05.
Epoch 65/1000
50000/50000 [==============================] - 6s 114us/step - loss: 0.0515 - acc: 0.9992 - val_loss: 0.1033 - val_acc: 0.9829

Epoch 00066: LearningRateScheduler reducing learning rate to 1.5151515151515151e-05.
Epoch 66/1000
50000/50000 [==============================] - 6s 129us/step - loss: 0.0510 - acc: 0.9993 - val_loss: 0.1034 - val_acc: 0.9830

Epoch 00067: LearningRateScheduler reducing learning rate to 1.4925373134328359e-05.
Epoch 67/1000
50000/50000 [==============================] - 6s 122us/step - loss: 0.0508 - acc: 0.9992 - val_loss: 0.1033 - val_acc: 0.9825
ubuntu@alexa-ml:~/0x06-keras$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x06-keras
File: 7-train.py
   
## 8. Save Only the Best

mandatory
Based on 7-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False): to also save the best iteration of the model:

save_best is a boolean indicating whether to save the model after each epoch if it is the best
a model is considered the best if its validation loss is the lowest that the model has obtained
filepath is the file path where the model should be saved
ubuntu@alexa-ml:~/0x06-keras$ cat 8-main.py 
#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model 


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)
    X_valid = datasets['X_valid']
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = datasets['Y_valid']
    Y_valid_oh = one_hot(Y_valid)

    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)
    batch_size = 64
    epochs = 1000
    train_model(network, X_train, Y_train_oh, batch_size, epochs,
                validation_data=(X_valid, Y_valid_oh), early_stopping=True,
                patience=3, learning_rate_decay=True, alpha=alpha,
                save_best=True, filepath='network1.h5')

ubuntu@alexa-ml:~/0x06-keras$ ./8-main.py 
Train on 50000 samples, validate on 10000 samples

Epoch 00001: LearningRateScheduler reducing learning rate to 0.001.
Epoch 1/1000
50000/50000 [==============================] - 8s 157us/step - loss: 0.3508 - acc: 0.9202 - val_loss: 0.2174 - val_acc: 0.9602

Epoch 00002: LearningRateScheduler reducing learning rate to 0.0005.
Epoch 2/1000
50000/50000 [==============================] - 8s 157us/step - loss: 0.1823 - acc: 0.9705 - val_loss: 0.1691 - val_acc: 0.9743

Epoch 00003: LearningRateScheduler reducing learning rate to 0.0003333333333333333.
Epoch 3/1000
50000/50000 [==============================] - 6s 127us/step - loss: 0.1481 - acc: 0.9795 - val_loss: 0.1563 - val_acc: 0.9769

...

Epoch 00064: LearningRateScheduler reducing learning rate to 1.5625e-05.
Epoch 64/1000
50000/50000 [==============================] - 7s 133us/step - loss: 0.0517 - acc: 0.9990 - val_loss: 0.1029 - val_acc: 0.9827

Epoch 00065: LearningRateScheduler reducing learning rate to 1.5384615384615384e-05.
Epoch 65/1000
50000/50000 [==============================] - 5s 109us/step - loss: 0.0515 - acc: 0.9992 - val_loss: 0.1033 - val_acc: 0.9829

Epoch 00066: LearningRateScheduler reducing learning rate to 1.5151515151515151e-05.
Epoch 66/1000
50000/50000 [==============================] - 6s 112us/step - loss: 0.0510 - acc: 0.9993 - val_loss: 0.1034 - val_acc: 0.9830

Epoch 00067: LearningRateScheduler reducing learning rate to 1.4925373134328359e-05.
Epoch 67/1000
50000/50000 [==============================] - 6s 114us/step - loss: 0.0508 - acc: 0.9992 - val_loss: 0.1033 - val_acc: 0.9825
ubuntu@alexa-ml:~/0x06-keras$ ls network1.h5 
network1.h5
ubuntu@alexa-ml:~/0x06-keras$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x06-keras
File: 8-train.py
   
## 9. Save and Load Model

mandatory
Write the following functions:

def save_model(network, filename): saves an entire model:
network is the model to save
filename is the path of the file that the model should be saved to
Returns: None
def load_model(filename): loads an entire model:
filename is the path of the file that the model should be loaded from
Returns: the loaded model
ubuntu@alexa-ml:~/0x06-keras$ cat 9-main.py
#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model 
model = __import__('9-model')

if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)
    X_valid = datasets['X_valid']
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = datasets['Y_valid']
    Y_valid_oh = one_hot(Y_valid)

    network = model.load_model('network1.h5')
    batch_size = 32
    epochs = 1000
    train_model(network, X_train, Y_train_oh, batch_size, epochs,
                validation_data=(X_valid, Y_valid_oh), early_stopping=True,
                patience=2, learning_rate_decay=True, alpha=0.001)
    model.save_model(network, 'network2.h5')
    network.summary()
    print(network.get_weights())
    del network

    network2 = model.load_model('network2.h5')
    network2.summary()
    print(network2.get_weights())

ubuntu@alexa-ml:~/0x06-keras$ ./9-main.py
Train on 50000 samples, validate on 10000 samples

Epoch 00001: LearningRateScheduler reducing learning rate to 0.001.
Epoch 1/1000
50000/50000 [==============================] - 11s 218us/step - loss: 0.1840 - acc: 0.9635 - val_loss: 0.1638 - val_acc: 0.9716

Epoch 00002: LearningRateScheduler reducing learning rate to 0.0005.
Epoch 2/1000
50000/50000 [==============================] - 9s 176us/step - loss: 0.1049 - acc: 0.9868 - val_loss: 0.1358 - val_acc: 0.9779

Epoch 00003: LearningRateScheduler reducing learning rate to 0.0003333333333333333.
Epoch 3/1000
50000/50000 [==============================] - 8s 165us/step - loss: 0.0834 - acc: 0.9920 - val_loss: 0.1216 - val_acc: 0.9808

Epoch 00004: LearningRateScheduler reducing learning rate to 0.00025.
Epoch 4/1000
50000/50000 [==============================] - 12s 231us/step - loss: 0.0729 - acc: 0.9946 - val_loss: 0.1180 - val_acc: 0.9808

Epoch 00005: LearningRateScheduler reducing learning rate to 0.0002.
Epoch 5/1000
50000/50000 [==============================] - 11s 221us/step - loss: 0.0664 - acc: 0.9959 - val_loss: 0.1144 - val_acc: 0.9809

Epoch 00006: LearningRateScheduler reducing learning rate to 0.00016666666666666666.
Epoch 6/1000
50000/50000 [==============================] - 9s 183us/step - loss: 0.0616 - acc: 0.9967 - val_loss: 0.1113 - val_acc: 0.9803

Epoch 00007: LearningRateScheduler reducing learning rate to 0.00014285714285714287.
Epoch 7/1000
50000/50000 [==============================] - 9s 189us/step - loss: 0.0583 - acc: 0.9973 - val_loss: 0.1045 - val_acc: 0.9821

Epoch 00008: LearningRateScheduler reducing learning rate to 0.000125.
Epoch 8/1000
50000/50000 [==============================] - 9s 178us/step - loss: 0.0556 - acc: 0.9976 - val_loss: 0.1047 - val_acc: 0.9822

Epoch 00009: LearningRateScheduler reducing learning rate to 0.00011111111111111112.
Epoch 9/1000
50000/50000 [==============================] - 8s 165us/step - loss: 0.0539 - acc: 0.9976 - val_loss: 0.1048 - val_acc: 0.9818
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               200960    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 269,322
Trainable params: 269,322
Non-trainable params: 0
_________________________________________________________________
[array([[-3.8009610e-32, -4.7635498e-33, -1.4057525e-32, ...,
         4.3825994e-32,  3.9422965e-32, -4.7520981e-32],
       [-5.0400801e-32, -2.0104553e-33,  8.8701746e-33, ...,
        -4.8622008e-32, -8.8164897e-33,  4.0189464e-32],
       [ 4.7300247e-33, -4.9894786e-32, -5.2076457e-32, ...,
        -5.0855447e-32,  3.8804655e-32,  6.2529952e-33],
       ...,
       [-8.7033484e-33, -5.2582877e-32, -5.3911492e-32, ...,
         3.9439422e-33, -4.0354157e-32, -4.7937788e-32],
       [-1.3583064e-32,  5.1161146e-32, -5.6233028e-32, ...,
         8.7784978e-33,  3.6725162e-33, -1.4458428e-32],
       [ 2.0569182e-32,  4.6331581e-32, -4.6222636e-32, ...,
         3.2422212e-32,  4.0120704e-32, -5.0064581e-32]], dtype=float32), array([-5.42255165e-03,  1.19785637e-01,  5.23720346e-02,  1.03110410e-02,
        1.87107902e-02,  1.41989086e-02,  1.12350071e-02,  2.76008509e-02,
        1.27801269e-01, -3.89967524e-02, -2.02556495e-02, -4.62714843e-02,
        1.30194187e-01,  4.29115910e-03,  6.65472075e-02,  1.54704798e-03,
       -5.88263869e-02,  8.44721347e-02, -7.35184038e-03,  4.62194085e-02,
        3.73042934e-02,  7.04525486e-02,  2.10950859e-02,  7.91235417e-02,
        7.28602335e-02,  4.59115058e-02, -4.63508964e-02, -2.21509635e-02,
       -9.16822348e-03,  2.80001163e-02, -9.89757702e-02,  1.47809163e-01,
        3.50833917e-03,  1.15984432e-01, -2.55682189e-02, -8.91123433e-03,
       -1.68071799e-02, -2.17670165e-02,  8.79516476e-04, -5.47891902e-03,
       -5.50239533e-03, -2.84654042e-03,  9.88868698e-02, -4.55969013e-02,
        5.75555861e-02, -1.90851800e-02, -1.68833062e-02,  1.11860596e-02,
        2.99566053e-03, -2.29080115e-02,  1.41166538e-01,  3.54361385e-02,
        6.52417261e-03, -5.97108006e-02, -2.56007351e-03, -1.01229943e-01,
       -4.44865040e-02, -5.73677383e-03, -3.99919115e-02, -7.33643100e-02,
        2.78245788e-02, -5.09251468e-02,  2.58603925e-03,  3.79152298e-02,
       -4.71243169e-03, -1.95143409e-02,  3.56299356e-02, -5.79533577e-02,
        3.44148278e-02,  2.16114670e-02,  5.39492331e-02, -2.11088993e-02,
        4.61111404e-02, -5.59479697e-03,  3.69911194e-02, -3.05724014e-02,
       -5.86141013e-02,  2.19059531e-02,  5.65099381e-02,  9.83161405e-02,
       -2.64573321e-02,  4.73500267e-02, -2.10616197e-02, -8.09149258e-03,
        7.01221526e-02, -8.24146345e-02, -2.27536038e-02,  8.21699053e-02,
        3.87316346e-02,  3.15262973e-02, -1.55786453e-02, -5.97577076e-03,
       -3.44991521e-03,  8.64902809e-02, -1.36852637e-01,  3.16732973e-02,
        3.33485985e-03, -8.63953400e-03,  3.88498604e-02, -1.10751055e-02,
        1.81695279e-02,  8.66026385e-04, -2.47641020e-02,  3.63605022e-02,
       -3.76436375e-02,  3.42167378e-03, -4.66135554e-02, -1.93273723e-02,
        8.97753909e-02,  4.01710495e-02, -4.11117077e-02,  1.07764497e-01,
       -1.32064419e-02, -4.27396968e-02, -9.60233063e-02, -7.99996108e-02,
        2.96253152e-02, -4.90622632e-02,  3.90885174e-02,  1.30973477e-03,
        1.53741958e-02, -6.30079210e-03,  5.70209697e-03,  7.69555792e-02,
        2.21333709e-02,  1.02476135e-01,  1.25097139e-02, -3.15427557e-02,
       -2.46393625e-02, -7.97851160e-02, -7.09855556e-02,  1.90447830e-02,
        2.13283929e-03, -6.36921972e-02, -4.66645993e-02,  5.66449650e-02,
        1.46611510e-02,  1.32085672e-02,  5.16292565e-02, -2.65292376e-02,
       -8.70395973e-02, -5.09263873e-02, -2.99491873e-03, -1.18522346e-02,
       -2.33896151e-02, -9.47765782e-02, -6.99510192e-03,  3.44670191e-02,
        6.14305376e-04, -6.04404137e-02,  1.86015554e-02,  1.94717478e-02,
       -2.19570827e-02,  6.75229682e-03, -5.86605668e-02,  4.13896563e-03,
       -1.85364615e-02, -5.13353273e-02, -7.16174170e-02, -6.04901910e-02,
       -3.97531176e-03, -1.02133378e-02,  3.16860452e-02, -6.21607937e-02,
        1.51735712e-02, -7.02523068e-02,  5.86631894e-02,  7.26487115e-02,
        1.96908526e-02, -9.02154855e-03, -3.47945876e-02,  1.32073721e-04,
       -3.64982523e-02, -4.22905870e-02,  5.41402623e-02,  7.54025131e-02,
       -7.98583701e-02,  1.07912138e-01,  1.15326587e-02, -7.30423536e-03,
        4.48674662e-03,  1.96290631e-02,  5.83971068e-02, -5.25176302e-02,
        8.11480638e-03, -4.88318363e-03,  3.93577442e-02,  6.15799241e-02,
        6.65078238e-02,  1.19150467e-02, -3.98552902e-02,  7.68825710e-02,
       -7.10767806e-02, -6.23493083e-02, -5.94618507e-02,  3.66897993e-02,
        3.27307358e-02,  8.74284282e-02, -9.85233858e-02,  2.11340957e-03,
        5.57199642e-02, -4.17726524e-02, -1.34163005e-02,  5.95312677e-02,
        3.20401639e-02,  2.10705698e-02,  1.62398592e-02, -6.38844445e-03,
        1.91176273e-02, -3.59788463e-02, -2.65758727e-02, -8.92021954e-02,
        8.81231204e-02, -2.64080055e-02, -1.80902034e-02, -2.24631783e-02,
        2.21795421e-02, -6.06609657e-02,  8.07675868e-02, -1.99074969e-02,
        6.06684387e-02, -5.86771555e-02,  8.72262791e-02, -1.27969065e-03,
       -2.53308658e-02, -1.54622477e-02, -4.27149534e-02, -3.23627405e-02,
       -6.79803044e-02,  1.44614819e-02,  1.08358823e-02,  1.14791922e-03,
       -4.92133833e-02, -1.77217070e-02,  7.63013819e-03,  7.35992426e-03,
       -7.65544921e-02, -2.70789402e-04,  9.02245194e-02,  8.17074254e-02,
       -2.74846703e-02, -6.42111748e-02, -7.52724567e-03,  7.78001593e-03,
       -3.14909369e-02, -3.74384485e-02,  7.60471588e-03, -2.43803486e-02,
        7.27141872e-02,  4.45889384e-02, -3.34395729e-02,  5.95564768e-02,
       -6.51311057e-05, -4.13893424e-02,  8.34922418e-02, -2.82471124e-02],
      dtype=float32), array([[-6.8193418e-03, -4.3311447e-02, -2.2587109e-02, ...,
        -1.3333004e-03,  1.1198742e-03,  7.1355708e-02],
       [-8.7768510e-03, -5.4443255e-02,  8.7477863e-03, ...,
         8.1711039e-03,  6.1310731e-02, -1.3987484e-02],
       [-3.4817703e-02,  2.8888771e-02,  3.0248387e-02, ...,
         7.4379356e-03,  6.5395519e-02, -2.6671611e-02],
       ...,
       [-2.4342051e-02, -4.4713490e-02, -3.8670462e-02, ...,
         5.1178265e-02,  8.1578113e-02, -1.1511453e-02],
       [ 7.8641780e-02,  2.7468946e-02, -2.0481899e-02, ...,
        -1.0952364e-02,  3.3160388e-02,  2.6741007e-03],
       [ 1.4569525e-06,  5.5528275e-05,  2.2682036e-05, ...,
        -6.8198424e-06, -3.0450277e-05,  2.2139466e-05]], dtype=float32), array([ 8.6902846e-03,  9.8969273e-02,  1.2555283e-01,  6.6254444e-02,
        1.1655674e-02,  1.5745793e-01,  1.3045560e-01,  5.5979639e-03,
       -2.6812542e-02,  1.4096253e-01, -7.3327363e-02,  6.6665865e-02,
       -6.0460209e-03,  6.3849390e-02, -9.6244805e-02, -1.1654791e-02,
        6.4250633e-02,  2.9507946e-02,  8.4809966e-02,  1.7389655e-01,
        6.2100779e-02,  5.9233098e-03,  1.5840566e-03, -1.2285662e-02,
        1.4297791e-01,  6.4667545e-02,  1.6712245e-02,  1.1615166e-01,
       -8.7909913e-03, -7.6202959e-02, -2.0751055e-02, -2.4535516e-02,
       -3.7224889e-03,  3.5019197e-02, -1.3351994e-02, -4.7939834e-03,
        9.5242016e-02,  3.5831314e-02,  1.0268044e-01,  1.6098891e-01,
        6.6069163e-02,  1.2056256e-01, -3.4086719e-02, -3.4383725e-02,
        5.1353619e-02,  2.8859915e-02,  1.6352838e-01,  4.1005641e-02,
        1.3431785e-01,  2.8951475e-03, -2.1281708e-02,  5.6862727e-02,
        1.0544681e-01,  1.5781955e-01,  7.7006176e-02, -1.9046010e-02,
        2.1722302e-02, -3.8828824e-03,  8.5470535e-02,  2.1235073e-02,
        3.9779365e-02, -9.1342732e-02, -7.1976468e-02,  1.3527745e-01,
        2.4901321e-02,  8.2063228e-02,  1.1289710e-01,  4.2514257e-02,
        3.0262357e-02, -1.5960310e-02, -2.3290550e-03, -5.2838501e-02,
        3.9216805e-02, -6.8539879e-03,  4.0055010e-02, -5.7410128e-02,
        1.2583114e-01, -2.3163997e-02,  8.0189429e-02,  1.2690836e-01,
       -6.2593110e-02,  7.2724514e-02,  2.0579065e-01,  1.5100455e-01,
        1.0570333e-01,  1.0547851e-01, -5.0013732e-02, -6.8121351e-02,
        1.3165326e-03,  8.5743308e-02,  2.5197692e-02,  4.1801356e-02,
        2.8855208e-02,  9.3787685e-02,  1.6236556e-01, -4.6338275e-02,
        1.3804600e-01,  1.3297901e-01,  2.3309229e-02,  5.5821408e-02,
        8.9127965e-02,  8.6506784e-02, -5.3759187e-02,  6.3532762e-02,
       -3.7570320e-02,  5.5799685e-02, -1.2883618e-02, -5.4759655e-02,
        3.3871844e-02,  1.3988510e-01,  1.4660260e-01,  1.5230735e-01,
        6.6068321e-02, -5.2617133e-02, -2.7143599e-02,  2.2198431e-02,
        5.8405701e-02,  8.4836043e-02, -8.1875592e-02,  4.7000919e-02,
        4.4559341e-02, -5.4258659e-02, -3.0311109e-02, -8.6952537e-02,
        1.0240083e-01,  2.4944553e-02,  2.8888699e-02, -1.0298248e-02,
       -3.6404099e-02,  1.9111334e-01, -3.2414519e-03,  1.3328564e-01,
        1.4363341e-01,  5.1877350e-02,  5.8149870e-02,  1.1324768e-01,
        8.5323930e-02,  9.3184009e-02,  7.8272335e-02,  7.5536355e-02,
        8.5016564e-03,  4.2266335e-02, -5.3507034e-02,  8.7683968e-02,
       -4.7006492e-02,  5.7886876e-02,  6.0920697e-02,  4.4110887e-02,
        9.6832372e-02,  1.5087926e-01, -9.3608452e-03,  1.7035361e-02,
        1.8508282e-01,  1.8660104e-01, -2.7490899e-02,  1.8379935e-05,
        1.7800622e-01, -6.7051053e-02, -4.6844989e-02, -9.3458369e-02,
        5.8709742e-03,  8.4132180e-02,  3.7760045e-02,  1.7084857e-03,
       -2.9528804e-02, -4.1546460e-02,  1.5757427e-01,  3.4335099e-02,
       -2.8841086e-02,  8.4256493e-02,  1.5563935e-01,  5.2509408e-02,
        1.5235119e-01,  8.3613679e-02, -4.8764922e-02,  9.3141254e-03,
        6.3053906e-02,  1.6177355e-01, -9.9167190e-03,  3.0503727e-02,
        3.2692537e-02,  3.6044732e-02,  2.9145889e-02, -8.3509553e-03,
       -2.5203347e-02, -6.1970230e-02,  3.3920057e-02,  1.0458338e-01,
        6.0746800e-02, -3.6521908e-02,  1.2550007e-01,  2.5732871e-02,
        5.8904346e-03,  9.2594571e-02,  1.7290029e-01, -7.8385279e-02,
        5.0777360e-03,  4.6757076e-02, -5.1762622e-02,  5.3370073e-02,
       -6.7386791e-02,  1.6365142e-01,  6.2478296e-03,  6.6207175e-04,
       -2.0671973e-02, -2.5592577e-02,  1.1633794e-01,  7.8679509e-02,
       -6.7169741e-02,  2.2829333e-02,  2.9602909e-02, -4.0508170e-02,
        1.7227107e-01,  4.1538954e-02,  1.9616881e-02,  6.6807643e-02,
        6.7487627e-02,  2.3291329e-02, -2.4825132e-02,  9.0276241e-02,
       -4.4160519e-02,  7.6820776e-02,  7.6962233e-02,  1.2671390e-01,
        1.5740655e-01,  5.9743752e-03,  7.4380517e-02,  6.6431858e-02,
        8.5746631e-02,  3.4872922e-03,  1.2870293e-02,  7.4270985e-04,
        2.3856882e-02,  4.1545838e-02,  3.2453869e-02,  8.4524810e-02,
       -5.5636819e-02,  1.5405205e-02, -5.3766835e-02, -1.2152072e-02,
        2.8115712e-02, -9.5226295e-02, -4.5381509e-02, -8.8090701e-03,
        1.8806040e-01,  8.9698568e-02,  7.9526193e-03,  2.9542342e-02,
        4.9743271e-03,  1.0985307e-01, -3.8334232e-02, -2.6247973e-02,
       -5.3359687e-02,  2.2953292e-02,  3.1281214e-02,  3.6531478e-02],
      dtype=float32), array([[-0.0594307 ,  0.19452482, -0.2823843 , ...,  0.06184928,
        -0.02412683, -0.29995787],
       [-0.20479676,  0.09350423, -0.11133969, ..., -0.14134312,
         0.21493568, -0.14635867],
       [ 0.165593  ,  0.01210315, -0.01291312, ..., -0.05217365,
         0.18512513, -0.13725156],
       ...,
       [ 0.03987962, -0.14602913,  0.01076543, ...,  0.06850222,
        -0.24221672, -0.05472106],
       [ 0.17652929,  0.15142111,  0.2597426 , ..., -0.07304035,
        -0.02805572,  0.09629773],
       [-0.13085422, -0.22056961,  0.05302332, ...,  0.06077137,
        -0.04070508,  0.2946609 ]], dtype=float32), array([ 0.00402458, -0.09855989, -0.01859134, -0.01681106, -0.0134734 ,
       -0.04443603, -0.05618203, -0.08323891,  0.20838276,  0.03688779],
      dtype=float32)]
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               200960    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 269,322
Trainable params: 269,322
Non-trainable params: 0
_________________________________________________________________
[array([[-3.8009610e-32, -4.7635498e-33, -1.4057525e-32, ...,
         4.3825994e-32,  3.9422965e-32, -4.7520981e-32],
       [-5.0400801e-32, -2.0104553e-33,  8.8701746e-33, ...,
        -4.8622008e-32, -8.8164897e-33,  4.0189464e-32],
       [ 4.7300247e-33, -4.9894786e-32, -5.2076457e-32, ...,
        -5.0855447e-32,  3.8804655e-32,  6.2529952e-33],
       ...,
       [-8.7033484e-33, -5.2582877e-32, -5.3911492e-32, ...,
         3.9439422e-33, -4.0354157e-32, -4.7937788e-32],
       [-1.3583064e-32,  5.1161146e-32, -5.6233028e-32, ...,
         8.7784978e-33,  3.6725162e-33, -1.4458428e-32],
       [ 2.0569182e-32,  4.6331581e-32, -4.6222636e-32, ...,
         3.2422212e-32,  4.0120704e-32, -5.0064581e-32]], dtype=float32), array([-5.42255165e-03,  1.19785637e-01,  5.23720346e-02,  1.03110410e-02,
        1.87107902e-02,  1.41989086e-02,  1.12350071e-02,  2.76008509e-02,
        1.27801269e-01, -3.89967524e-02, -2.02556495e-02, -4.62714843e-02,
        1.30194187e-01,  4.29115910e-03,  6.65472075e-02,  1.54704798e-03,
       -5.88263869e-02,  8.44721347e-02, -7.35184038e-03,  4.62194085e-02,
        3.73042934e-02,  7.04525486e-02,  2.10950859e-02,  7.91235417e-02,
        7.28602335e-02,  4.59115058e-02, -4.63508964e-02, -2.21509635e-02,
       -9.16822348e-03,  2.80001163e-02, -9.89757702e-02,  1.47809163e-01,
        3.50833917e-03,  1.15984432e-01, -2.55682189e-02, -8.91123433e-03,
       -1.68071799e-02, -2.17670165e-02,  8.79516476e-04, -5.47891902e-03,
       -5.50239533e-03, -2.84654042e-03,  9.88868698e-02, -4.55969013e-02,
        5.75555861e-02, -1.90851800e-02, -1.68833062e-02,  1.11860596e-02,
        2.99566053e-03, -2.29080115e-02,  1.41166538e-01,  3.54361385e-02,
        6.52417261e-03, -5.97108006e-02, -2.56007351e-03, -1.01229943e-01,
       -4.44865040e-02, -5.73677383e-03, -3.99919115e-02, -7.33643100e-02,
        2.78245788e-02, -5.09251468e-02,  2.58603925e-03,  3.79152298e-02,
       -4.71243169e-03, -1.95143409e-02,  3.56299356e-02, -5.79533577e-02,
        3.44148278e-02,  2.16114670e-02,  5.39492331e-02, -2.11088993e-02,
        4.61111404e-02, -5.59479697e-03,  3.69911194e-02, -3.05724014e-02,
       -5.86141013e-02,  2.19059531e-02,  5.65099381e-02,  9.83161405e-02,
       -2.64573321e-02,  4.73500267e-02, -2.10616197e-02, -8.09149258e-03,
        7.01221526e-02, -8.24146345e-02, -2.27536038e-02,  8.21699053e-02,
        3.87316346e-02,  3.15262973e-02, -1.55786453e-02, -5.97577076e-03,
       -3.44991521e-03,  8.64902809e-02, -1.36852637e-01,  3.16732973e-02,
        3.33485985e-03, -8.63953400e-03,  3.88498604e-02, -1.10751055e-02,
        1.81695279e-02,  8.66026385e-04, -2.47641020e-02,  3.63605022e-02,
       -3.76436375e-02,  3.42167378e-03, -4.66135554e-02, -1.93273723e-02,
        8.97753909e-02,  4.01710495e-02, -4.11117077e-02,  1.07764497e-01,
       -1.32064419e-02, -4.27396968e-02, -9.60233063e-02, -7.99996108e-02,
        2.96253152e-02, -4.90622632e-02,  3.90885174e-02,  1.30973477e-03,
        1.53741958e-02, -6.30079210e-03,  5.70209697e-03,  7.69555792e-02,
        2.21333709e-02,  1.02476135e-01,  1.25097139e-02, -3.15427557e-02,
       -2.46393625e-02, -7.97851160e-02, -7.09855556e-02,  1.90447830e-02,
        2.13283929e-03, -6.36921972e-02, -4.66645993e-02,  5.66449650e-02,
        1.46611510e-02,  1.32085672e-02,  5.16292565e-02, -2.65292376e-02,
       -8.70395973e-02, -5.09263873e-02, -2.99491873e-03, -1.18522346e-02,
       -2.33896151e-02, -9.47765782e-02, -6.99510192e-03,  3.44670191e-02,
        6.14305376e-04, -6.04404137e-02,  1.86015554e-02,  1.94717478e-02,
       -2.19570827e-02,  6.75229682e-03, -5.86605668e-02,  4.13896563e-03,
       -1.85364615e-02, -5.13353273e-02, -7.16174170e-02, -6.04901910e-02,
       -3.97531176e-03, -1.02133378e-02,  3.16860452e-02, -6.21607937e-02,
        1.51735712e-02, -7.02523068e-02,  5.86631894e-02,  7.26487115e-02,
        1.96908526e-02, -9.02154855e-03, -3.47945876e-02,  1.32073721e-04,
       -3.64982523e-02, -4.22905870e-02,  5.41402623e-02,  7.54025131e-02,
       -7.98583701e-02,  1.07912138e-01,  1.15326587e-02, -7.30423536e-03,
        4.48674662e-03,  1.96290631e-02,  5.83971068e-02, -5.25176302e-02,
        8.11480638e-03, -4.88318363e-03,  3.93577442e-02,  6.15799241e-02,
        6.65078238e-02,  1.19150467e-02, -3.98552902e-02,  7.68825710e-02,
       -7.10767806e-02, -6.23493083e-02, -5.94618507e-02,  3.66897993e-02,
        3.27307358e-02,  8.74284282e-02, -9.85233858e-02,  2.11340957e-03,
        5.57199642e-02, -4.17726524e-02, -1.34163005e-02,  5.95312677e-02,
        3.20401639e-02,  2.10705698e-02,  1.62398592e-02, -6.38844445e-03,
        1.91176273e-02, -3.59788463e-02, -2.65758727e-02, -8.92021954e-02,
        8.81231204e-02, -2.64080055e-02, -1.80902034e-02, -2.24631783e-02,
        2.21795421e-02, -6.06609657e-02,  8.07675868e-02, -1.99074969e-02,
        6.06684387e-02, -5.86771555e-02,  8.72262791e-02, -1.27969065e-03,
       -2.53308658e-02, -1.54622477e-02, -4.27149534e-02, -3.23627405e-02,
       -6.79803044e-02,  1.44614819e-02,  1.08358823e-02,  1.14791922e-03,
       -4.92133833e-02, -1.77217070e-02,  7.63013819e-03,  7.35992426e-03,
       -7.65544921e-02, -2.70789402e-04,  9.02245194e-02,  8.17074254e-02,
       -2.74846703e-02, -6.42111748e-02, -7.52724567e-03,  7.78001593e-03,
       -3.14909369e-02, -3.74384485e-02,  7.60471588e-03, -2.43803486e-02,
        7.27141872e-02,  4.45889384e-02, -3.34395729e-02,  5.95564768e-02,
       -6.51311057e-05, -4.13893424e-02,  8.34922418e-02, -2.82471124e-02],
      dtype=float32), array([[-6.8193418e-03, -4.3311447e-02, -2.2587109e-02, ...,
        -1.3333004e-03,  1.1198742e-03,  7.1355708e-02],
       [-8.7768510e-03, -5.4443255e-02,  8.7477863e-03, ...,
         8.1711039e-03,  6.1310731e-02, -1.3987484e-02],
       [-3.4817703e-02,  2.8888771e-02,  3.0248387e-02, ...,
         7.4379356e-03,  6.5395519e-02, -2.6671611e-02],
       ...,
       [-2.4342051e-02, -4.4713490e-02, -3.8670462e-02, ...,
         5.1178265e-02,  8.1578113e-02, -1.1511453e-02],
       [ 7.8641780e-02,  2.7468946e-02, -2.0481899e-02, ...,
        -1.0952364e-02,  3.3160388e-02,  2.6741007e-03],
       [ 1.4569525e-06,  5.5528275e-05,  2.2682036e-05, ...,
        -6.8198424e-06, -3.0450277e-05,  2.2139466e-05]], dtype=float32), array([ 8.6902846e-03,  9.8969273e-02,  1.2555283e-01,  6.6254444e-02,
        1.1655674e-02,  1.5745793e-01,  1.3045560e-01,  5.5979639e-03,
       -2.6812542e-02,  1.4096253e-01, -7.3327363e-02,  6.6665865e-02,
       -6.0460209e-03,  6.3849390e-02, -9.6244805e-02, -1.1654791e-02,
        6.4250633e-02,  2.9507946e-02,  8.4809966e-02,  1.7389655e-01,
        6.2100779e-02,  5.9233098e-03,  1.5840566e-03, -1.2285662e-02,
        1.4297791e-01,  6.4667545e-02,  1.6712245e-02,  1.1615166e-01,
       -8.7909913e-03, -7.6202959e-02, -2.0751055e-02, -2.4535516e-02,
       -3.7224889e-03,  3.5019197e-02, -1.3351994e-02, -4.7939834e-03,
        9.5242016e-02,  3.5831314e-02,  1.0268044e-01,  1.6098891e-01,
        6.6069163e-02,  1.2056256e-01, -3.4086719e-02, -3.4383725e-02,
        5.1353619e-02,  2.8859915e-02,  1.6352838e-01,  4.1005641e-02,
        1.3431785e-01,  2.8951475e-03, -2.1281708e-02,  5.6862727e-02,
        1.0544681e-01,  1.5781955e-01,  7.7006176e-02, -1.9046010e-02,
        2.1722302e-02, -3.8828824e-03,  8.5470535e-02,  2.1235073e-02,
        3.9779365e-02, -9.1342732e-02, -7.1976468e-02,  1.3527745e-01,
        2.4901321e-02,  8.2063228e-02,  1.1289710e-01,  4.2514257e-02,
        3.0262357e-02, -1.5960310e-02, -2.3290550e-03, -5.2838501e-02,
        3.9216805e-02, -6.8539879e-03,  4.0055010e-02, -5.7410128e-02,
        1.2583114e-01, -2.3163997e-02,  8.0189429e-02,  1.2690836e-01,
       -6.2593110e-02,  7.2724514e-02,  2.0579065e-01,  1.5100455e-01,
        1.0570333e-01,  1.0547851e-01, -5.0013732e-02, -6.8121351e-02,
        1.3165326e-03,  8.5743308e-02,  2.5197692e-02,  4.1801356e-02,
        2.8855208e-02,  9.3787685e-02,  1.6236556e-01, -4.6338275e-02,
        1.3804600e-01,  1.3297901e-01,  2.3309229e-02,  5.5821408e-02,
        8.9127965e-02,  8.6506784e-02, -5.3759187e-02,  6.3532762e-02,
       -3.7570320e-02,  5.5799685e-02, -1.2883618e-02, -5.4759655e-02,
        3.3871844e-02,  1.3988510e-01,  1.4660260e-01,  1.5230735e-01,
        6.6068321e-02, -5.2617133e-02, -2.7143599e-02,  2.2198431e-02,
        5.8405701e-02,  8.4836043e-02, -8.1875592e-02,  4.7000919e-02,
        4.4559341e-02, -5.4258659e-02, -3.0311109e-02, -8.6952537e-02,
        1.0240083e-01,  2.4944553e-02,  2.8888699e-02, -1.0298248e-02,
       -3.6404099e-02,  1.9111334e-01, -3.2414519e-03,  1.3328564e-01,
        1.4363341e-01,  5.1877350e-02,  5.8149870e-02,  1.1324768e-01,
        8.5323930e-02,  9.3184009e-02,  7.8272335e-02,  7.5536355e-02,
        8.5016564e-03,  4.2266335e-02, -5.3507034e-02,  8.7683968e-02,
       -4.7006492e-02,  5.7886876e-02,  6.0920697e-02,  4.4110887e-02,
        9.6832372e-02,  1.5087926e-01, -9.3608452e-03,  1.7035361e-02,
        1.8508282e-01,  1.8660104e-01, -2.7490899e-02,  1.8379935e-05,
        1.7800622e-01, -6.7051053e-02, -4.6844989e-02, -9.3458369e-02,
        5.8709742e-03,  8.4132180e-02,  3.7760045e-02,  1.7084857e-03,
       -2.9528804e-02, -4.1546460e-02,  1.5757427e-01,  3.4335099e-02,
       -2.8841086e-02,  8.4256493e-02,  1.5563935e-01,  5.2509408e-02,
        1.5235119e-01,  8.3613679e-02, -4.8764922e-02,  9.3141254e-03,
        6.3053906e-02,  1.6177355e-01, -9.9167190e-03,  3.0503727e-02,
        3.2692537e-02,  3.6044732e-02,  2.9145889e-02, -8.3509553e-03,
       -2.5203347e-02, -6.1970230e-02,  3.3920057e-02,  1.0458338e-01,
        6.0746800e-02, -3.6521908e-02,  1.2550007e-01,  2.5732871e-02,
        5.8904346e-03,  9.2594571e-02,  1.7290029e-01, -7.8385279e-02,
        5.0777360e-03,  4.6757076e-02, -5.1762622e-02,  5.3370073e-02,
       -6.7386791e-02,  1.6365142e-01,  6.2478296e-03,  6.6207175e-04,
       -2.0671973e-02, -2.5592577e-02,  1.1633794e-01,  7.8679509e-02,
       -6.7169741e-02,  2.2829333e-02,  2.9602909e-02, -4.0508170e-02,
        1.7227107e-01,  4.1538954e-02,  1.9616881e-02,  6.6807643e-02,
        6.7487627e-02,  2.3291329e-02, -2.4825132e-02,  9.0276241e-02,
       -4.4160519e-02,  7.6820776e-02,  7.6962233e-02,  1.2671390e-01,
        1.5740655e-01,  5.9743752e-03,  7.4380517e-02,  6.6431858e-02,
        8.5746631e-02,  3.4872922e-03,  1.2870293e-02,  7.4270985e-04,
        2.3856882e-02,  4.1545838e-02,  3.2453869e-02,  8.4524810e-02,
       -5.5636819e-02,  1.5405205e-02, -5.3766835e-02, -1.2152072e-02,
        2.8115712e-02, -9.5226295e-02, -4.5381509e-02, -8.8090701e-03,
        1.8806040e-01,  8.9698568e-02,  7.9526193e-03,  2.9542342e-02,
        4.9743271e-03,  1.0985307e-01, -3.8334232e-02, -2.6247973e-02,
       -5.3359687e-02,  2.2953292e-02,  3.1281214e-02,  3.6531478e-02],
      dtype=float32), array([[-0.0594307 ,  0.19452482, -0.2823843 , ...,  0.06184928,
        -0.02412683, -0.29995787],
       [-0.20479676,  0.09350423, -0.11133969, ..., -0.14134312,
         0.21493568, -0.14635867],
       [ 0.165593  ,  0.01210315, -0.01291312, ..., -0.05217365,
         0.18512513, -0.13725156],
       ...,
       [ 0.03987962, -0.14602913,  0.01076543, ...,  0.06850222,
        -0.24221672, -0.05472106],
       [ 0.17652929,  0.15142111,  0.2597426 , ..., -0.07304035,
        -0.02805572,  0.09629773],
       [-0.13085422, -0.22056961,  0.05302332, ...,  0.06077137,
        -0.04070508,  0.2946609 ]], dtype=float32), array([ 0.00402458, -0.09855989, -0.01859134, -0.01681106, -0.0134734 ,
       -0.04443603, -0.05618203, -0.08323891,  0.20838276,  0.03688779],
      dtype=float32)]
ubuntu@alexa-ml:~/0x06-keras$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x06-keras
File: 9-model.py
   
## 10. Save and Load Weights

mandatory
Write the following functions:

def save_weights(network, filename, save_format='h5'): saves a model’s weights:
network is the model whose weights should be saved
filename is the path of the file that the weights should be saved to
save_format is the format in which the weights should be saved
Returns: None
def load_weights(network, filename): loads a model’s weights:
network is the model to which the weights should be loaded
filename is the path of the file that the weights should be loaded from
Returns: None
ubuntu@alexa-ml:~/0x06-keras$ cat 10-main.py
#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model
model = __import__('9-model')
weights = __import__('10-weights')

if __name__ == '__main__':

    network = model.load_model('network2.h5')
    weights.save_weights(network, 'weights2.h5')
    del network

    network2 = model.load_model('network1.h5')
    print(network2.get_weights())
    weights.load_weights(network2, 'weights2.h5')
    print(network2.get_weights())

ubuntu@alexa-ml:~/0x06-keras$ ./10-main.py
[array([[-3.8009610e-32,  1.0071884e-31, -1.1896290e-31, ...,
         4.3825994e-32,  9.6112162e-32, -7.7076594e-32],
       [-8.1747245e-32, -2.0104553e-33,  1.1594371e-31, ...,
        -7.8862194e-32, -1.1524185e-31,  4.0189464e-32],
       [-1.0001117e-31, -8.0926662e-32, -8.4465329e-32, ...,
        -5.0855447e-32,  3.8804655e-32, -1.3102230e-31],
       ...,
       [-1.1376338e-31, -6.6505056e-32, -5.3911492e-32, ...,
         3.9439422e-33, -9.8382318e-32, -7.0399910e-32],
       [-1.3583064e-32,  8.2980603e-32, -5.6233028e-32, ...,
         1.1474513e-31,  3.6725162e-33, -1.2235531e-31],
       [ 2.0569182e-32,  8.9056757e-32, -7.1609388e-32, ...,
         3.2422212e-32,  9.7813267e-32, -6.7512690e-32]], dtype=float32), array([ 1.58284046e-02,  7.50938877e-02,  5.55931628e-02,  9.96411312e-03,
        5.28343837e-04,  2.56864559e-02,  3.63821723e-02,  3.22434679e-02,
        8.66339952e-02, -2.32772548e-02, -1.65830664e-02, -5.11219865e-03,
        8.57980177e-02,  3.59634832e-02,  4.91884761e-02,  6.45830482e-03,
       -2.06307080e-02,  5.06407395e-02,  7.06305553e-04,  2.78945155e-02,
        4.31745872e-02,  5.43350875e-02,  4.56395075e-02,  4.81381156e-02,
        3.75241823e-02,  4.00379300e-02, -1.75596457e-02, -1.48604037e-02,
        1.58586856e-02,  2.98569631e-02, -2.41514761e-02,  1.05141826e-01,
        1.13808811e-02,  6.27070218e-02, -3.52727162e-04,  1.46074910e-02,
       -5.81867248e-03,  1.22419139e-02,  8.24793149e-03,  1.83175653e-02,
        3.11604775e-02,  2.96896249e-02,  7.14566931e-02, -9.25473683e-03,
        5.47560416e-02, -2.48050736e-03,  1.38276974e-02,  1.02478713e-02,
        1.33312624e-02,  7.76342209e-03,  7.96581954e-02,  2.75195315e-02,
        6.37142882e-02, -3.09232604e-02,  1.58507545e-02, -4.49547246e-02,
       -3.12114018e-03,  2.03015208e-02, -2.61817072e-02, -1.76353045e-02,
        4.56345566e-02, -3.68254818e-02,  1.69123877e-02,  4.57399748e-02,
        2.25385018e-02, -9.44158249e-03,  3.50949019e-02, -2.10165512e-02,
        3.75490822e-02,  1.52684646e-02,  6.51115701e-02, -1.19293341e-02,
        4.36716713e-02, -1.00409947e-02,  4.65573445e-02, -2.10097134e-02,
       -1.51595846e-02,  2.99567133e-02,  4.85033505e-02,  8.86763260e-02,
        1.91397257e-02,  2.88771465e-03,  4.28518606e-03, -6.76579494e-03,
        3.75263728e-02, -1.94955040e-02,  2.75979415e-02,  4.82166111e-02,
        5.15641049e-02,  2.82720756e-02,  4.96803550e-04,  2.71842093e-03,
        8.65454413e-03,  5.86728267e-02, -5.81379421e-02,  3.67067978e-02,
        1.05254129e-02, -4.10144497e-03,  2.42255218e-02,  1.84916910e-02,
        3.55863906e-02,  5.32900961e-03,  7.50796823e-03,  3.65532413e-02,
        1.37308147e-03,  3.67517793e-03, -1.06394906e-02,  2.95443125e-02,
        6.76082000e-02,  3.31847966e-02, -6.39715698e-03,  5.49406819e-02,
       -1.23150581e-02, -1.94825996e-02, -3.35792154e-02, -4.19720709e-02,
        1.35854110e-02, -5.13689928e-02,  3.32078189e-02,  5.95856085e-03,
        2.72434466e-02, -3.46258539e-03, -5.59390243e-03,  5.74971139e-02,
       -5.29011618e-03,  5.04215993e-02,  3.98309268e-02, -1.01160286e-02,
        1.43442797e-02, -4.02778387e-02, -2.80474406e-02,  2.77178884e-02,
        2.71978672e-03, -4.33270857e-02, -1.67728625e-02,  4.64957133e-02,
        4.85485792e-02,  3.45048532e-02,  1.03393897e-01, -4.26865509e-03,
       -1.84873231e-02, -2.29788087e-02,  2.65704393e-02, -1.07107535e-02,
        5.24958086e-05, -3.02482098e-02,  1.82033908e-02,  4.98788282e-02,
       -1.36658491e-03, -4.09213342e-02,  9.28086787e-03,  7.36255059e-03,
       -6.22688793e-03,  1.91584732e-02, -3.44326012e-02,  2.10269988e-02,
       -1.58591603e-03, -1.96915865e-02, -3.55102234e-02, -1.76052451e-02,
        3.21833342e-02,  8.08821176e-04,  9.16090887e-03, -2.21153162e-02,
       -1.53940823e-02, -3.90307158e-02,  2.62597203e-02,  5.60990199e-02,
        3.82002033e-02,  1.32315718e-02, -3.98718519e-03,  2.41657104e-02,
       -9.37329081e-04,  2.40398310e-02,  2.22620275e-02,  5.13032041e-02,
       -3.05405464e-02,  8.78388286e-02,  3.44196483e-02,  3.06636803e-02,
        2.67141070e-02,  5.61922342e-02,  4.24447432e-02, -9.12650861e-03,
        5.75380847e-02,  4.28869836e-02,  4.48621660e-02,  6.11533821e-02,
        5.95438890e-02,  1.93128549e-02, -1.47372205e-02,  7.99189433e-02,
       -4.88573015e-02, -3.81754301e-02, -1.64362732e-02,  1.90840233e-02,
        5.94151504e-02,  6.27147406e-02, -4.22051363e-02, -4.17308789e-03,
        3.89469340e-02,  2.53811481e-06, -1.34163005e-02,  3.72273996e-02,
        7.10768625e-02,  1.48481755e-02, -6.68292167e-03,  3.21253203e-02,
        3.65632363e-02, -1.51157957e-02, -9.73089784e-03, -3.49751227e-02,
        6.47157207e-02, -8.97444598e-03,  1.52834635e-02,  1.49770547e-02,
        4.60935496e-02, -3.61642276e-04,  5.95332310e-02, -1.16354432e-02,
        4.38150950e-02, -1.76979396e-02,  6.12880848e-02,  7.02325022e-03,
       -4.33601532e-03, -1.03334896e-02, -3.47818509e-02,  1.79491658e-02,
       -3.39301042e-02,  3.12849060e-02,  9.59425175e-04,  1.94651224e-02,
        1.83663275e-02,  1.88396070e-02,  7.60967564e-03,  1.68920364e-02,
       -2.10571568e-02,  1.11699607e-02,  7.00308904e-02,  5.79195656e-02,
       -9.73488484e-03, -4.04580906e-02,  3.14609483e-02, -3.15647776e-04,
       -4.65753954e-03, -2.21004561e-02,  1.11513678e-02, -1.03611760e-02,
        4.55683172e-02,  6.63453043e-02,  1.17711127e-02,  4.22723331e-02,
        3.75862271e-02, -1.39793139e-02,  5.53907752e-02,  8.83557834e-03],
      dtype=float32), array([[-0.00371919, -0.05007912, -0.01812696, ..., -0.01382338,
         0.02415221,  0.0925543 ],
       [ 0.02203175, -0.03561179,  0.03445467, ..., -0.01496624,
         0.04926918, -0.01780702],
       [-0.0337771 ,  0.10885999,  0.02123713, ...,  0.03437595,
         0.0522139 , -0.08206959],
       ...,
       [-0.0626606 , -0.00720833, -0.03656305, ...,  0.020274  ,
         0.07797025,  0.00332946],
       [-0.02517798,  0.0218024 , -0.01239559, ..., -0.00100349,
         0.06393236,  0.01268309],
       [ 0.0289655 , -0.03470461,  0.02073021, ..., -0.00241219,
         0.06299865, -0.05209595]], dtype=float32), array([ 0.04652996,  0.043997  ,  0.07114699,  0.03903542,  0.02767882,
        0.0863481 ,  0.07621165,  0.00214916, -0.02681254,  0.05791308,
       -0.00147518,  0.04299389,  0.02798902,  0.04657304, -0.0244261 ,
        0.02415524,  0.02945878,  0.02993501,  0.05815235,  0.08068836,
        0.05819618,  0.01217864,  0.01892883,  0.00516294,  0.05656728,
        0.02579783,  0.02820423,  0.06847014,  0.01176183, -0.02039377,
       -0.02075106,  0.0430186 ,  0.00511082,  0.04038307,  0.01102713,
        0.02655444,  0.03633076,  0.04672799,  0.06999496,  0.08631742,
        0.05746292,  0.04565633,  0.00201734,  0.00250425,  0.04414648,
        0.030997  ,  0.06914084,  0.0351956 ,  0.06476128,  0.01695583,
        0.02081329,  0.0532925 ,  0.03754437,  0.05675277,  0.04702865,
        0.00759455,  0.02971335,  0.02909641,  0.03495037,  0.03537805,
        0.02035745, -0.04563862, -0.00351922,  0.07011741,  0.0196888 ,
        0.03890587,  0.06129541,  0.05956952,  0.03319855,  0.00379073,
        0.0111421 , -0.00077544,  0.0257228 ,  0.01889134,  0.03754475,
       -0.02212741,  0.05560778,  0.00536041,  0.04807876,  0.06203355,
        0.00561534,  0.07013745,  0.09062373,  0.0670809 ,  0.07314177,
        0.0570873 ,  0.00371935, -0.0207575 ,  0.04322195,  0.0582999 ,
        0.02316695,  0.04613077,  0.02948793,  0.05750891,  0.04428294,
        0.01313044,  0.07786941,  0.0835509 ,  0.02538908,  0.04790534,
        0.06006417,  0.06522411,  0.01149572,  0.06367876,  0.00614   ,
        0.02801823,  0.01319215, -0.01029942,  0.03345457,  0.07364164,
        0.06301042,  0.07558368,  0.08211317, -0.01943968,  0.01806634,
        0.01951652,  0.05525405,  0.06553435, -0.02302993,  0.01267116,
        0.0583568 , -0.01412742, -0.02275325, -0.03249238,  0.05950009,
       -0.00165116,  0.03237157,  0.02667695, -0.00054109,  0.07375826,
        0.03655118,  0.05314804,  0.07384741,  0.04892747,  0.0648926 ,
        0.06137131,  0.04840928,  0.04782393,  0.05954469,  0.05557283,
        0.01695545,  0.01749786, -0.01102989,  0.04035386, -0.00065627,
        0.04278678,  0.03946338,  0.0356142 ,  0.06702255,  0.06879313,
        0.0070997 ,  0.02651379,  0.05873521,  0.0831332 , -0.004174  ,
        0.01542924,  0.08181876, -0.01126117, -0.00763193, -0.02914163,
        0.03213413,  0.07089414,  0.00016282,  0.02037768, -0.00567811,
       -0.00296772,  0.06790668,  0.02571199,  0.00207151,  0.04883849,
        0.0799915 ,  0.05242474,  0.07621986,  0.04209178, -0.0148869 ,
        0.00974209,  0.02491404,  0.08366543,  0.02592914,  0.03094619,
        0.04194048,  0.01911422,  0.0276555 ,  0.02067538,  0.01583027,
       -0.00321123,  0.0403896 ,  0.06466128,  0.04367998,  0.00593152,
        0.06596841,  0.00846652,  0.03917108,  0.05471751,  0.07656774,
       -0.01729305,  0.01655496,  0.0343636 , -0.00146066,  0.06529658,
       -0.01639492,  0.1009846 ,  0.03078729,  0.00892706, -0.00317575,
        0.01336057,  0.05209352,  0.03555475, -0.00106674,  0.01427909,
        0.0523075 ,  0.01873922,  0.08049697,  0.03650277,  0.04813057,
        0.05196897,  0.05204869,  0.03833751,  0.01386643,  0.0628586 ,
       -0.02861879,  0.05983486,  0.03318779,  0.06697322,  0.0611463 ,
        0.01547751,  0.0444758 ,  0.03945206,  0.06331456,  0.02524176,
        0.01701661,  0.01745347,  0.02651639,  0.03222635,  0.01924249,
        0.04716564, -0.0182713 ,  0.0460152 ,  0.00583924,  0.02017109,
        0.04373844, -0.03440706,  0.01394549,  0.00593979,  0.08492176,
        0.05958203,  0.00500921,  0.03021434,  0.02667685,  0.05597633,
       -0.01744312,  0.01281162,  0.00459726,  0.04171734,  0.02743316,
        0.02273934], dtype=float32), array([[-0.06505688,  0.20432228, -0.3115043 , ...,  0.01855138,
        -0.05560027, -0.25403774],
       [-0.22421178,  0.10835828, -0.07947999, ..., -0.15718825,
         0.17537546, -0.16371952],
       [ 0.14985822, -0.02830164,  0.05024177, ..., -0.0465285 ,
         0.18274635, -0.10761911],
       ...,
       [ 0.06473439, -0.11770464,  0.01381425, ...,  0.03004427,
        -0.2598251 , -0.05676018],
       [ 0.19455092,  0.13349947,  0.23735559, ..., -0.0543741 ,
         0.0149371 ,  0.11647129],
       [-0.13190398, -0.20505469,  0.07300279, ...,  0.04788497,
        -0.07708195,  0.31805626]], dtype=float32), array([ 0.00375419, -0.02424374, -0.00884842, -0.01260086,  0.01054102,
       -0.02130446, -0.0192898 , -0.026981  ,  0.06677306,  0.00853687],
      dtype=float32)]
[array([[-3.8009610e-32, -4.7635498e-33, -1.4057525e-32, ...,
         4.3825994e-32,  3.9422965e-32, -4.7520981e-32],
       [-5.0400801e-32, -2.0104553e-33,  8.8701746e-33, ...,
        -4.8622008e-32, -8.8164897e-33,  4.0189464e-32],
       [ 4.7300247e-33, -4.9894786e-32, -5.2076457e-32, ...,
        -5.0855447e-32,  3.8804655e-32,  6.2529952e-33],
       ...,
       [-8.7033484e-33, -5.2582877e-32, -5.3911492e-32, ...,
         3.9439422e-33, -4.0354157e-32, -4.7937788e-32],
       [-1.3583064e-32,  5.1161146e-32, -5.6233028e-32, ...,
         8.7784978e-33,  3.6725162e-33, -1.4458428e-32],
       [ 2.0569182e-32,  4.6331581e-32, -4.6222636e-32, ...,
         3.2422212e-32,  4.0120704e-32, -5.0064581e-32]], dtype=float32), array([-5.42255165e-03,  1.19785637e-01,  5.23720346e-02,  1.03110410e-02,
        1.87107902e-02,  1.41989086e-02,  1.12350071e-02,  2.76008509e-02,
        1.27801269e-01, -3.89967524e-02, -2.02556495e-02, -4.62714843e-02,
        1.30194187e-01,  4.29115910e-03,  6.65472075e-02,  1.54704798e-03,
       -5.88263869e-02,  8.44721347e-02, -7.35184038e-03,  4.62194085e-02,
        3.73042934e-02,  7.04525486e-02,  2.10950859e-02,  7.91235417e-02,
        7.28602335e-02,  4.59115058e-02, -4.63508964e-02, -2.21509635e-02,
       -9.16822348e-03,  2.80001163e-02, -9.89757702e-02,  1.47809163e-01,
        3.50833917e-03,  1.15984432e-01, -2.55682189e-02, -8.91123433e-03,
       -1.68071799e-02, -2.17670165e-02,  8.79516476e-04, -5.47891902e-03,
       -5.50239533e-03, -2.84654042e-03,  9.88868698e-02, -4.55969013e-02,
        5.75555861e-02, -1.90851800e-02, -1.68833062e-02,  1.11860596e-02,
        2.99566053e-03, -2.29080115e-02,  1.41166538e-01,  3.54361385e-02,
        6.52417261e-03, -5.97108006e-02, -2.56007351e-03, -1.01229943e-01,
       -4.44865040e-02, -5.73677383e-03, -3.99919115e-02, -7.33643100e-02,
        2.78245788e-02, -5.09251468e-02,  2.58603925e-03,  3.79152298e-02,
       -4.71243169e-03, -1.95143409e-02,  3.56299356e-02, -5.79533577e-02,
        3.44148278e-02,  2.16114670e-02,  5.39492331e-02, -2.11088993e-02,
        4.61111404e-02, -5.59479697e-03,  3.69911194e-02, -3.05724014e-02,
       -5.86141013e-02,  2.19059531e-02,  5.65099381e-02,  9.83161405e-02,
       -2.64573321e-02,  4.73500267e-02, -2.10616197e-02, -8.09149258e-03,
        7.01221526e-02, -8.24146345e-02, -2.27536038e-02,  8.21699053e-02,
        3.87316346e-02,  3.15262973e-02, -1.55786453e-02, -5.97577076e-03,
       -3.44991521e-03,  8.64902809e-02, -1.36852637e-01,  3.16732973e-02,
        3.33485985e-03, -8.63953400e-03,  3.88498604e-02, -1.10751055e-02,
        1.81695279e-02,  8.66026385e-04, -2.47641020e-02,  3.63605022e-02,
       -3.76436375e-02,  3.42167378e-03, -4.66135554e-02, -1.93273723e-02,
        8.97753909e-02,  4.01710495e-02, -4.11117077e-02,  1.07764497e-01,
       -1.32064419e-02, -4.27396968e-02, -9.60233063e-02, -7.99996108e-02,
        2.96253152e-02, -4.90622632e-02,  3.90885174e-02,  1.30973477e-03,
        1.53741958e-02, -6.30079210e-03,  5.70209697e-03,  7.69555792e-02,
        2.21333709e-02,  1.02476135e-01,  1.25097139e-02, -3.15427557e-02,
       -2.46393625e-02, -7.97851160e-02, -7.09855556e-02,  1.90447830e-02,
        2.13283929e-03, -6.36921972e-02, -4.66645993e-02,  5.66449650e-02,
        1.46611510e-02,  1.32085672e-02,  5.16292565e-02, -2.65292376e-02,
       -8.70395973e-02, -5.09263873e-02, -2.99491873e-03, -1.18522346e-02,
       -2.33896151e-02, -9.47765782e-02, -6.99510192e-03,  3.44670191e-02,
        6.14305376e-04, -6.04404137e-02,  1.86015554e-02,  1.94717478e-02,
       -2.19570827e-02,  6.75229682e-03, -5.86605668e-02,  4.13896563e-03,
       -1.85364615e-02, -5.13353273e-02, -7.16174170e-02, -6.04901910e-02,
       -3.97531176e-03, -1.02133378e-02,  3.16860452e-02, -6.21607937e-02,
        1.51735712e-02, -7.02523068e-02,  5.86631894e-02,  7.26487115e-02,
        1.96908526e-02, -9.02154855e-03, -3.47945876e-02,  1.32073721e-04,
       -3.64982523e-02, -4.22905870e-02,  5.41402623e-02,  7.54025131e-02,
       -7.98583701e-02,  1.07912138e-01,  1.15326587e-02, -7.30423536e-03,
        4.48674662e-03,  1.96290631e-02,  5.83971068e-02, -5.25176302e-02,
        8.11480638e-03, -4.88318363e-03,  3.93577442e-02,  6.15799241e-02,
        6.65078238e-02,  1.19150467e-02, -3.98552902e-02,  7.68825710e-02,
       -7.10767806e-02, -6.23493083e-02, -5.94618507e-02,  3.66897993e-02,
        3.27307358e-02,  8.74284282e-02, -9.85233858e-02,  2.11340957e-03,
        5.57199642e-02, -4.17726524e-02, -1.34163005e-02,  5.95312677e-02,
        3.20401639e-02,  2.10705698e-02,  1.62398592e-02, -6.38844445e-03,
        1.91176273e-02, -3.59788463e-02, -2.65758727e-02, -8.92021954e-02,
        8.81231204e-02, -2.64080055e-02, -1.80902034e-02, -2.24631783e-02,
        2.21795421e-02, -6.06609657e-02,  8.07675868e-02, -1.99074969e-02,
        6.06684387e-02, -5.86771555e-02,  8.72262791e-02, -1.27969065e-03,
       -2.53308658e-02, -1.54622477e-02, -4.27149534e-02, -3.23627405e-02,
       -6.79803044e-02,  1.44614819e-02,  1.08358823e-02,  1.14791922e-03,
       -4.92133833e-02, -1.77217070e-02,  7.63013819e-03,  7.35992426e-03,
       -7.65544921e-02, -2.70789402e-04,  9.02245194e-02,  8.17074254e-02,
       -2.74846703e-02, -6.42111748e-02, -7.52724567e-03,  7.78001593e-03,
       -3.14909369e-02, -3.74384485e-02,  7.60471588e-03, -2.43803486e-02,
        7.27141872e-02,  4.45889384e-02, -3.34395729e-02,  5.95564768e-02,
       -6.51311057e-05, -4.13893424e-02,  8.34922418e-02, -2.82471124e-02],
      dtype=float32), array([[-6.8193418e-03, -4.3311447e-02, -2.2587109e-02, ...,
        -1.3333004e-03,  1.1198742e-03,  7.1355708e-02],
       [-8.7768510e-03, -5.4443255e-02,  8.7477863e-03, ...,
         8.1711039e-03,  6.1310731e-02, -1.3987484e-02],
       [-3.4817703e-02,  2.8888771e-02,  3.0248387e-02, ...,
         7.4379356e-03,  6.5395519e-02, -2.6671611e-02],
       ...,
       [-2.4342051e-02, -4.4713490e-02, -3.8670462e-02, ...,
         5.1178265e-02,  8.1578113e-02, -1.1511453e-02],
       [ 7.8641780e-02,  2.7468946e-02, -2.0481899e-02, ...,
        -1.0952364e-02,  3.3160388e-02,  2.6741007e-03],
       [ 1.4569525e-06,  5.5528275e-05,  2.2682036e-05, ...,
        -6.8198424e-06, -3.0450277e-05,  2.2139466e-05]], dtype=float32), array([ 8.6902846e-03,  9.8969273e-02,  1.2555283e-01,  6.6254444e-02,
        1.1655674e-02,  1.5745793e-01,  1.3045560e-01,  5.5979639e-03,
       -2.6812542e-02,  1.4096253e-01, -7.3327363e-02,  6.6665865e-02,
       -6.0460209e-03,  6.3849390e-02, -9.6244805e-02, -1.1654791e-02,
        6.4250633e-02,  2.9507946e-02,  8.4809966e-02,  1.7389655e-01,
        6.2100779e-02,  5.9233098e-03,  1.5840566e-03, -1.2285662e-02,
        1.4297791e-01,  6.4667545e-02,  1.6712245e-02,  1.1615166e-01,
       -8.7909913e-03, -7.6202959e-02, -2.0751055e-02, -2.4535516e-02,
       -3.7224889e-03,  3.5019197e-02, -1.3351994e-02, -4.7939834e-03,
        9.5242016e-02,  3.5831314e-02,  1.0268044e-01,  1.6098891e-01,
        6.6069163e-02,  1.2056256e-01, -3.4086719e-02, -3.4383725e-02,
        5.1353619e-02,  2.8859915e-02,  1.6352838e-01,  4.1005641e-02,
        1.3431785e-01,  2.8951475e-03, -2.1281708e-02,  5.6862727e-02,
        1.0544681e-01,  1.5781955e-01,  7.7006176e-02, -1.9046010e-02,
        2.1722302e-02, -3.8828824e-03,  8.5470535e-02,  2.1235073e-02,
        3.9779365e-02, -9.1342732e-02, -7.1976468e-02,  1.3527745e-01,
        2.4901321e-02,  8.2063228e-02,  1.1289710e-01,  4.2514257e-02,
        3.0262357e-02, -1.5960310e-02, -2.3290550e-03, -5.2838501e-02,
        3.9216805e-02, -6.8539879e-03,  4.0055010e-02, -5.7410128e-02,
        1.2583114e-01, -2.3163997e-02,  8.0189429e-02,  1.2690836e-01,
       -6.2593110e-02,  7.2724514e-02,  2.0579065e-01,  1.5100455e-01,
        1.0570333e-01,  1.0547851e-01, -5.0013732e-02, -6.8121351e-02,
        1.3165326e-03,  8.5743308e-02,  2.5197692e-02,  4.1801356e-02,
        2.8855208e-02,  9.3787685e-02,  1.6236556e-01, -4.6338275e-02,
        1.3804600e-01,  1.3297901e-01,  2.3309229e-02,  5.5821408e-02,
        8.9127965e-02,  8.6506784e-02, -5.3759187e-02,  6.3532762e-02,
       -3.7570320e-02,  5.5799685e-02, -1.2883618e-02, -5.4759655e-02,
        3.3871844e-02,  1.3988510e-01,  1.4660260e-01,  1.5230735e-01,
        6.6068321e-02, -5.2617133e-02, -2.7143599e-02,  2.2198431e-02,
        5.8405701e-02,  8.4836043e-02, -8.1875592e-02,  4.7000919e-02,
        4.4559341e-02, -5.4258659e-02, -3.0311109e-02, -8.6952537e-02,
        1.0240083e-01,  2.4944553e-02,  2.8888699e-02, -1.0298248e-02,
       -3.6404099e-02,  1.9111334e-01, -3.2414519e-03,  1.3328564e-01,
        1.4363341e-01,  5.1877350e-02,  5.8149870e-02,  1.1324768e-01,
        8.5323930e-02,  9.3184009e-02,  7.8272335e-02,  7.5536355e-02,
        8.5016564e-03,  4.2266335e-02, -5.3507034e-02,  8.7683968e-02,
       -4.7006492e-02,  5.7886876e-02,  6.0920697e-02,  4.4110887e-02,
        9.6832372e-02,  1.5087926e-01, -9.3608452e-03,  1.7035361e-02,
        1.8508282e-01,  1.8660104e-01, -2.7490899e-02,  1.8379935e-05,
        1.7800622e-01, -6.7051053e-02, -4.6844989e-02, -9.3458369e-02,
        5.8709742e-03,  8.4132180e-02,  3.7760045e-02,  1.7084857e-03,
       -2.9528804e-02, -4.1546460e-02,  1.5757427e-01,  3.4335099e-02,
       -2.8841086e-02,  8.4256493e-02,  1.5563935e-01,  5.2509408e-02,
        1.5235119e-01,  8.3613679e-02, -4.8764922e-02,  9.3141254e-03,
        6.3053906e-02,  1.6177355e-01, -9.9167190e-03,  3.0503727e-02,
        3.2692537e-02,  3.6044732e-02,  2.9145889e-02, -8.3509553e-03,
       -2.5203347e-02, -6.1970230e-02,  3.3920057e-02,  1.0458338e-01,
        6.0746800e-02, -3.6521908e-02,  1.2550007e-01,  2.5732871e-02,
        5.8904346e-03,  9.2594571e-02,  1.7290029e-01, -7.8385279e-02,
        5.0777360e-03,  4.6757076e-02, -5.1762622e-02,  5.3370073e-02,
       -6.7386791e-02,  1.6365142e-01,  6.2478296e-03,  6.6207175e-04,
       -2.0671973e-02, -2.5592577e-02,  1.1633794e-01,  7.8679509e-02,
       -6.7169741e-02,  2.2829333e-02,  2.9602909e-02, -4.0508170e-02,
        1.7227107e-01,  4.1538954e-02,  1.9616881e-02,  6.6807643e-02,
        6.7487627e-02,  2.3291329e-02, -2.4825132e-02,  9.0276241e-02,
       -4.4160519e-02,  7.6820776e-02,  7.6962233e-02,  1.2671390e-01,
        1.5740655e-01,  5.9743752e-03,  7.4380517e-02,  6.6431858e-02,
        8.5746631e-02,  3.4872922e-03,  1.2870293e-02,  7.4270985e-04,
        2.3856882e-02,  4.1545838e-02,  3.2453869e-02,  8.4524810e-02,
       -5.5636819e-02,  1.5405205e-02, -5.3766835e-02, -1.2152072e-02,
        2.8115712e-02, -9.5226295e-02, -4.5381509e-02, -8.8090701e-03,
        1.8806040e-01,  8.9698568e-02,  7.9526193e-03,  2.9542342e-02,
        4.9743271e-03,  1.0985307e-01, -3.8334232e-02, -2.6247973e-02,
       -5.3359687e-02,  2.2953292e-02,  3.1281214e-02,  3.6531478e-02],
      dtype=float32), array([[-0.0594307 ,  0.19452482, -0.2823843 , ...,  0.06184928,
        -0.02412683, -0.29995787],
       [-0.20479676,  0.09350423, -0.11133969, ..., -0.14134312,
         0.21493568, -0.14635867],
       [ 0.165593  ,  0.01210315, -0.01291312, ..., -0.05217365,
         0.18512513, -0.13725156],
       ...,
       [ 0.03987962, -0.14602913,  0.01076543, ...,  0.06850222,
        -0.24221672, -0.05472106],
       [ 0.17652929,  0.15142111,  0.2597426 , ..., -0.07304035,
        -0.02805572,  0.09629773],
       [-0.13085422, -0.22056961,  0.05302332, ...,  0.06077137,
        -0.04070508,  0.2946609 ]], dtype=float32), array([ 0.00402458, -0.09855989, -0.01859134, -0.01681106, -0.0134734 ,
       -0.04443603, -0.05618203, -0.08323891,  0.20838276,  0.03688779],
      dtype=float32)]
ubuntu@alexa-ml:~/0x06-keras$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x06-keras
File: 10-weights.py
   
## 11. Save and Load Configuration

mandatory
Write the following functions:

def save_config(network, filename): saves a model’s configuration in JSON format:
network is the model whose configuration should be saved
filename is the path of the file that the configuration should be saved to
Returns: None
def load_config(filename): loads a model with a specific configuration:
filename is the path of the file containing the model’s configuration in JSON format
Returns: the loaded model
ubuntu@alexa-ml:~/0x06-keras$ cat 11-main.py
#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
model = __import__('9-model')
config = __import__('11-config')

if __name__ == '__main__':
    network = model.load_model('network1.h5')
    config.save_config(network, 'config1.json')
    del network

    network2 = config.load_config('config1.json')
    network2.summary()
    print(network2.get_weights())

ubuntu@alexa-ml:~/0x06-keras$ ./11-main.py
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               200960    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 269,322
Trainable params: 269,322
Non-trainable params: 0
_________________________________________________________________
[array([[ 0.07595653,  0.09581027, -0.01175209, ..., -0.05243838,
         0.04540078, -0.09269386],
       [-0.01323028, -0.051954  , -0.01268669, ...,  0.00432736,
         0.03686089, -0.07104349],
       [-0.00924175, -0.04997446,  0.0242543 , ..., -0.06823482,
         0.05516547,  0.03175139],
       ...,
       [ 0.03273007, -0.04632739,  0.03379987, ..., -0.07104938,
        -0.05403581, -0.03537126],
       [ 0.09671515,  0.01242327,  0.08824161, ...,  0.00318845,
        -0.09294248,  0.00738481],
       [ 0.02152885,  0.01395665,  0.0101698 , ..., -0.00165461,
        -0.04027275, -0.00877788]], dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0.], dtype=float32), array([[-0.06833467, -0.03180679,  0.00837077, ..., -0.1575552 ,
        -0.05222473, -0.13664919],
       [-0.05603951, -0.09797473,  0.00573276, ...,  0.16201185,
         0.10563677,  0.08692238],
       [ 0.0773556 , -0.07601337,  0.04726284, ..., -0.00312303,
         0.07468981, -0.11122718],
       ...,
       [-0.09624373, -0.03031957,  0.05009373, ...,  0.11220471,
        -0.12641405, -0.15056057],
       [ 0.07753017, -0.04575136, -0.06678326, ...,  0.03294286,
        -0.10902938, -0.08459996],
       [ 0.01357522, -0.07630654, -0.08225919, ...,  0.08785751,
        -0.07642032, -0.01332911]], dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0.], dtype=float32), array([[-0.06897541, -0.16557665, -0.04771167, ...,  0.01455407,
         0.03382928, -0.17569515],
       [-0.05053294,  0.09438621,  0.11519638, ..., -0.01245164,
        -0.0719116 , -0.18455806],
       [ 0.09228224,  0.14074004,  0.06882233, ...,  0.05615992,
        -0.15130006,  0.02174817],
       ...,
       [ 0.00889782, -0.00705951,  0.04887312, ..., -0.08805028,
        -0.14918058, -0.1591385 ],
       [-0.14299504, -0.10059351, -0.10517051, ..., -0.06911735,
        -0.09655877,  0.04620347],
       [-0.16582027, -0.08827206,  0.16611351, ...,  0.01500075,
        -0.19330625, -0.11692349]], dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)]
ubuntu@alexa-ml:~/0x06-keras$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x06-keras
File: 11-config.py
   
## 12. Test

mandatory
Write a function def test_model(network, data, labels, verbose=True): that tests a neural network:

network is the network model to test
data is the input data to test the model with
labels are the correct one-hot labels of data
verbose is a boolean that determines if output should be printed during the testing process
Returns: the loss and accuracy of the model with the testing data, respectively
ubuntu@alexa-ml:~/0x06-keras$ cat 12-main.py 
#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot
load_model = __import__('9-model').load_model
test_model = __import__('12-test').test_model


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_test = datasets['X_test']
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = datasets['Y_test']
    Y_test_oh = one_hot(Y_test)

    network = load_model('network2.h5')
    print(test_model(network, X_test, Y_test_oh))
ubuntu@alexa-ml:~/0x06-keras$ ./12-main.py 
10000/10000 [==============================] - 0s 43us/step
[0.09121923210024833, 0.9832]
ubuntu@alexa-ml:~/0x06-keras$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x06-keras
File: 12-test.py
   
## 13. Predict

mandatory
Write a function def predict(network, data, verbose=False): that makes a prediction using a neural network:

network is the network model to make the prediction with
data is the input data to make the prediction with
verbose is a boolean that determines if output should be printed during the prediction process
Returns: the prediction for the data
ubuntu@alexa-ml:~/0x06-keras$ cat 13-main.py
#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot
load_model = __import__('9-model').load_model
predict = __import__('13-predict').predict


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_test = datasets['X_test']
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = datasets['Y_test']

    network = load_model('network2.h5')
    Y_pred = predict(network, X_test)
    print(Y_pred)
    print(np.argmax(Y_pred, axis=1))
    print(Y_test)
ubuntu@alexa-ml:~/0x06-keras$ ./13-main.py
2018-11-30 21:13:04.692277: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
[[1.09882777e-07 1.85020565e-06 7.01209501e-07 ... 9.99942422e-01
  2.60075751e-07 8.19494835e-06]
 [1.37503928e-08 1.84829651e-06 9.99997258e-01 ... 2.15385221e-09
  8.63893135e-09 8.08128995e-14]
 [1.03242555e-05 9.99097943e-01 1.67965060e-04 ... 5.23889903e-04
  7.54134162e-05 1.10524084e-07]
 ...
 [1.88145090e-11 5.88180065e-08 1.43965796e-12 ... 3.95040814e-07
  1.28503856e-08 2.26610467e-07]
 [2.37400890e-08 2.48911092e-09 1.20860308e-10 ... 1.69956849e-08
  5.97703838e-05 3.89016153e-10]
 [2.68221925e-08 1.28844213e-10 5.13091347e-09 ... 1.14895975e-11
  1.83396942e-09 7.46730282e-12]]
[7 2 1 ... 4 5 6]
[7 2 1 ... 4 5 6]
ubuntu@alexa-ml:~/0x06-keras$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x06-keras
File: 13-predict.py