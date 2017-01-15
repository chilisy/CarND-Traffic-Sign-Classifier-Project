import pickle, os
import numpy as np
import pandas
import zipfile
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle

cwd = os.getcwd()
data_dir = '../traffic-signs-data'
training_zipfile = cwd + data_dir + '/train.zip'

if 'train.p' not in os.listdir(cwd + data_dir):
    with zipfile.ZipFile(training_zipfile) as z:
        z.extract('train.p', cwd + data_dir)

training_file = cwd + data_dir + '/train.p'
testing_file = cwd + data_dir + '/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
X_sizes = train['sizes']
X_coords = train['coords']


