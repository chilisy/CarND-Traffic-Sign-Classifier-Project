import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import pandas


def get_signname_from_index(index):
    csv_file = '../signnames.csv'
    col_names = ['ClassId', 'SignName']
    data = pandas.read_csv(csv_file, names=col_names)
    signnames = data.SignName.tolist()
    signnames = signnames[1:len(signnames)]

    return signnames[index]


def conv2d(inputs, weights, bias, stride=2, padding='VALID'):
    strides = [1, stride, stride, 1]
    conv = tf.nn.conv2d(inputs, weights, strides, padding)
    out = tf.nn.bias_add(conv, bias)
    return out


def activation(features, opt='relu'):
    if opt == 'relu':
        return tf.nn.relu(features)
    elif opt == 'sigmoid':
        return tf.nn.sigmoid(features)
    elif opt == 'tanh':
        return tf.nn.tanh(features)


def maxpool2d(inputs, ksize, stride, padding='VALID'):
    ksizes = [1, ksize, ksize, 1]
    strides = [1, stride, stride, 1]

    out = tf.nn.max_pool(inputs, ksizes, strides, padding)
    return out


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    layer1_weights = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
    layer1_bias = tf.Variable(tf.zeros(6))
    layer1 = conv2d(x, layer1_weights, layer1_bias, stride=1)

    # Activation.
    layer1 = activation(layer1, opt='relu')

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    pool1 = maxpool2d(layer1, 2, 2)

    # Layer 2: Convolutional. Output = 10x10x16.
    layer2_weights = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    layer2_bias = tf.Variable(tf.zeros(16))
    layer2 = conv2d(pool1, layer2_weights, layer2_bias, stride=1)

    # Activation.
    layer2 = activation(layer2, opt='relu')

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool2 = maxpool2d(layer2, 2, 2)

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(pool2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_weights = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_bias = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_weights) + fc1_bias

    # Activation.
    fc1 = activation(fc1, opt='relu')

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_weights = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_bias = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_weights) + fc2_bias

    # Activation.
    fc2 = activation(fc2, opt='relu')

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_weights = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_bias = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_weights) + fc3_bias

    return logits


def optimize(eta = 0.001, opt='Adam'):
    if opt == 'Adam':
        return tf.train.AdamOptimizer(learning_rate=eta)
    elif opt == 'Adagrad':
        return tf.train.AdagradOptimizer(learning_rate=eta)
    elif opt == 'Momentum':
        return tf.train.MomentumOptimizer(learning_rate=eta)
    elif opt == 'AdagradDA':
        return tf.train.AdagradDAOptimizer(learning_rate=eta)
    elif opt == 'AdadeltaOptimizer':
        return tf.train.AdadeltaOptimizer(learning_rate=eta)

