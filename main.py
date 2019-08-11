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


np.random.seed(21052711)
limitador = None  # None for complete data

if __name__ == "__main__":
    tag = 'prueba_100'

    print('Reading paths to audiofiles, started at {}'.format(datetime.datetime.now()))
    audiofiles = [os.path.join(train_path, f) for f in listdir(train_path) if isfile(join(train_path, f))]
    reduxfiles = [os.path.join(train_redux_path, f)
                  for f in listdir(train_redux_path) if isfile(join(train_redux_path, f))]

    labels_dict = get_labels(labels_path)

    # Get a random permutation to remove ordering bias
    np.random.shuffle(audiofiles)
    np.random.shuffle(reduxfiles)

    # convert the list of files to numpy arrays and select a random smaller version if limitador is not None
    X_path = np.array(audiofiles)[:limitador]
    X_redux_path = np.array(reduxfiles)[:limitador]

    # TODO: not all audio clip in the redux dataset has the same length
    # from DataManager.Audio import read_aiff
    # print([(i.split('/')[-1][-5], read_aiff(i).shape[0]) for i in X_redux_path if read_aiff(i).shape[0] != 4000])
    # print([i for i in X_path if read_aiff(i).shape[0] != 4000])

    # Start the process of data extraction, spectrogram transformation and data enhancement
    print('Generating train and test split')
    X_train_path, X_test_path = train_test_split(np.concatenate([X_path, X_redux_path]), test_size=0.3)

    print('Getting test spectrograms')
    X_test, Y_test = get_spects(X_test_path, labels_dict)

    print('Getting train spectrograms + enhancement')
    X_train, Y_train = get_spects_enhanced(X_train_path, labels_dict)

    print(X_train.shape)
    #
    # print('Getting even more data adding noise to whale calls')
    # X_enhanced, Y_enhanced = enhance_with_noise(X_train, Y_train)
    #
    # print(Y_train)
    # print(Y_train)
    #
    # print(X_train.shape, X_enhanced.shape, Y_train.shape, Y_enhanced.shape)
    #
    # X_train, Y_train = np.concatenate([X_train, X_enhanced]), np.concatenate([Y_train, Y_enhanced])
