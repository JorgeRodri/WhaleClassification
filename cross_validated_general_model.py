# python basic pacakages
import aifc
import os
import datetime
from os import listdir
from os.path import isfile, join
import csv
import time

# required packags
import tensorflow as tf
import numpy as np
import pandas as pd

from matplotlib import mlab
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# my packages
from DataManager.Audio import *
from DataManager.General import *


def f_max(x):
    return x / x.max()


def log_norm(x):
    log_spect = np.log(x + 1)
    return log_spect / log_spect.max()


def f_05(x):
    x_min = x.min()
    return (x - x_min) / (x.max() - x_min) - .5


def f_norm(x):
    return (x - x.mean()) / x.std()


def preprocess(files_train, files_test, labels, normalization_function, to_categorical=True):

    x_test, y_test = get_spects(files_test, labels)

    x_train, y_train = get_spects_enhanced(files_train, labels)
    x_enhanced, y_enhanced = enhance_with_noise(x_train, y_train)

    x_train, y_train = np.concatenate([x_train, x_enhanced]), np.concatenate([y_train, y_enhanced])

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

    if to_categorical:
        y_train, y_test = tf.keras.utils.to_categorical(y_train, 2), tf.keras.utils.to_categorical(y_test, 2)
    else:
        y_train, y_test = y_train.astype(int), y_test.astype(int)

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    x_train = np.array(list(map(normalization_function, x_train)))
    x_test = np.array(list(map(normalization_function, x_test)))

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    return x_train, y_train, x_test, y_test


def k_fold_cross_validation(x_files, labels_path, normalization_f, n_splits=5, shuffle=True, seed=None):

    if seed is not None:
        np.random.seed(seed)
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    else:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)

    cvscores = []
    cvhistories = []
    cvauc = []

    opt = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    labels_dict = get_labels(labels_path)

    for train, test in kfold.split(x_files,  list(map(lambda x: labels_dict[x.split('/')[-1]], x_files))):

        x_train, y_train, x_test, y_test = preprocess(x_files[train], x_files[test], labels_dict, normalization_f)
        print('Test: Whale %.4f, Not Whale %.4f' % ((y_test == '1').sum() / y_test.shape[0],
                                                    (y_test == '0').sum() / y_test.shape[0]))

        print('Train: Whale %.4f, Not Whale %.4f' % ((y_train == '1').sum() / y_train.shape[0],
                                                     (y_train == '0').sum() / y_train.shape[0]))

        # Compile model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(20, kernel_size=(7, 7), activation=tf.nn.relu,
                                   input_shape=x_train.shape[1:], name='Conv1'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(40, kernel_size=(7, 7), activation=tf.nn.relu, name='Conv2'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=tf.nn.relu, name="Dense1"),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax, name="Softmax")
        ])
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=5, verbose=2, validation_split=0.2)
        score = model.evaluate(x_test, y_test)

        # Compute probabilities
        Y_pred = model.predict(x_test)
        # Assign most probable label
        y = np.argmax(y_test, axis=1)
        y_hat = np.argmax(Y_pred, axis=1)
        # Plot statistics
        print('Analysis of results')
        target_names = ['no_whale', 'whale']
        print(classification_report(y, y_hat, target_names=target_names))
        print(confusion_matrix(y, y_hat))

        y_pred_proba = model.predict(x_test)[:, 1]
        auc_roc = roc_auc_score(y, y_pred_proba)
        print('\nROC AUC with probabilities %.4f' % auc_roc)

        cvhistories.append(history)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
        cvscores.append(score[1] * 100)
        cvauc.append(auc_roc)

    return cvscores, cvhistories, cvauc


"""
DATA PATHS
"""
train_redux_path = "data/train2/"
labels_path = "data/train.csv"
train_path = "data/train/"

save_path = "result_graphs/"

"""
NAME FOR THE SAVED DOCUMENTS
"""
tag = 'example_training_{}'.format(time.time())

np.random.seed(21052711)  # Fixed random seed, change for data shuffle and test split

normalize = log_norm  # normalization function f_max, log_norm, f_05 and f_norm

optimizer_params = dict(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

if __name__ == "__main__":

    print('Reading paths to audiofiles, started at {}'.format(datetime.datetime.now()))
    audiofiles = np.array([os.path.join(train_path, f)
                           for f in listdir(train_path) if isfile(join(train_path, f))])[:1000]

    np.random.shuffle(audiofiles)

    score, history, aucs = k_fold_cross_validation(audiofiles, labels_path, normalize)

    print('Accuracy Scores: ', score)
    print('Accuracy Mean: ', np.mean(score))

    print('AUC ROC Scores: ', score)
    print('AUC ROC Mean: ', np.mean(aucs))
