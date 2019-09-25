# python basic pacakages
import aifc
import os
import datetime
from os import listdir
from os.path import isfile, join
import csv

# required packags
import tensorflow as tf
import numpy as np
import pandas as pd

from matplotlib import mlab
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# my packages
from DataManager.Audio import *
from DataManager.General import *


def preprocess(files_train, files_test, labels):

    x_test, y_test = get_spects(files_test, labels)

    x_train, y_train = get_spects_enhanced(files_train, labels)
    x_enhanced, y_enhanced = enhance_with_noise(x_train, y_train)

    x_train, y_train = np.concatenate([x_train, x_enhanced]), np.concatenate([y_train, y_enhanced])

    y_train, y_train = y_train.astype(int), y_test.astype(int)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

    y_train, y_test = tf.keras.utils.to_categorical(y_train, 2), tf.keras.utils.to_categorical(y_test, 2)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    return x_train, y_train, x_test, y_test


def k_fold_cross_validation(x_files, labels_path, model, f_preprocess, n_splits=5, shuffle=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    else:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)

    cvscores = []
    cvhistories = []

    opt = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    labels_dict = get_labels(labels_path)

    for train, test in kfold.split(x_files,  map(lambda x: labels_dict[x], x_files)):
        x_train, y_train, x_test, y_test = preprocess(x_files[train], x_files[test], labels_dict)
        # Compile model
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        history = model.model.fit(x_train, y_train, epochs=150, verbose=2, validation_split=0.2)
        score = model.model.evaluate(x_test, y_test)
        cvhistories.append(history)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
        cvscores.append(score[1] * 100)
