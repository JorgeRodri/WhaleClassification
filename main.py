import os
import time
import datetime

import numpy as np
from os.path import isfile, join
from sklearn.model_selection import train_test_split

from DataManager.Audio import get_spects, get_spects_enhanced
from DataManager.General import get_labels, enhance_with_noise

import tensorflow as tf
from os import listdir

import matplotlib.pyplot as plt

train_redux_path = "data/train2/"
labels_path = "data/train.csv"
train_path = "data/train/"
tag = 'prueba_100'

if __name__ == "__main__":

    print('Reading paths to audiofiles, started at {}'.format(datetime.datetime.now()))

    audiofiles = [os.path.join(train_path, f) for f in listdir(train_path) if isfile(join(train_path, f))]

    # TODO: [f[-5] for f in audiofiles][:10]

    np.random.shuffle(audiofiles)
    X_path = np.array(audiofiles)  # limitador

    print('Generating train and test split')
    X_train_path, X_test_path = train_test_split(X_path, test_size=0.3)

    print('Getting test spectrograms')
    X_test, Y_test = get_spects(X_test_path, labels_dict, cut=False)

    print('Getting train spectrograms + enhancement')
    X_train, Y_train = get_spects_enhanced(X_train_path, labels_dict, cut=False)

    print('Getting even more data adding noise to whale calls')
    X_enhanced, Y_enhanced = enhance_with_noise(X_train, Y_train)
    X_train, Y_train = np.concatenate([X_train, X_enhanced]), np.concatenate([Y_train, Y_enhanced])
