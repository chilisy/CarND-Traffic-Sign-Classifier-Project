import pickle, os
import numpy as np
import tensorflow as tf
from nn_functions import *
import zipfile
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle


cwd = os.getcwd()
data_dir = '/../traffic-signs-data'
training_zipfile = cwd + data_dir + '/train.zip'

# unzip train.zip if pickle-file not exist
if 'train.p' not in os.listdir(cwd + data_dir):
    with zipfile.ZipFile(training_zipfile) as z:
        z.extract('train.p', cwd + data_dir)

training_file = cwd + data_dir + '/train.p'
testing_file = cwd + data_dir + '/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train_raw, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
X_sizes = train['sizes']
X_coords = train['coords']

# Number of training examples
n_train = len(X_train_raw)

# Number of testing examples.
n_test = len(X_test)

# What's the shape of an traffic sign image?
image_shape = X_train_raw[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

# randomize data set
X_train_raw, X_sizes, X_coords, y_train = shuffle(X_train_raw, X_sizes, X_coords, y_train)

n_validation = int(np.ceil(0.15*n_train))
# print(n_validation, n_train)

X_validation = X_train_raw[0:n_validation]
X_train = X_train_raw[n_validation:n_train]
y_validation = y_train[0:n_validation]
y_train = y_train[n_validation:n_train]

# plot
showPlot = False
if showPlot:
    # Get random sign
    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()

    plt.figure()
    plt.imshow(image)
    plt.show()
    print(get_signname_from_index(y_train[index]))

EPOCHS = 200
BATCH_SIZE = 128

x = tf.placeholder(tf.float32, (None,) + image_shape)
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

learning_rate = 0.0008

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = optimize(eta = learning_rate, opt='Adam')
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.5f}".format(validation_accuracy))
        print()

    saver.save(sess, 'lenet')
    print("Model saved")


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

