import pickle, os
import numpy as np
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

# randomize data set
X_train, X_sizes, X_coords, y_train = shuffle(X_train_raw, X_sizes, X_coords, y_train)

# Plot
showPlot = False
if showPlot:
    # Get random sign
    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()

    plt.figure()
    plt.imshow(image)
    plt.show()
    print(get_signname_from_index(y_train[index]))



