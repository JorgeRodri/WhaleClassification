# python basic pacakages
import aifc
import os
import datetime
from os import listdir
from os.path import isfile, join
import csv

# required packags
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D, dot, BatchNormalization
from keras.callbacks import Callback
from keras.optimizers import RMSprop, Adam, SGD
from keras import regularizers

import numpy as np
import pandas as pd

from matplotlib import mlab
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# my packages
from DataManager.Audio import *
from DataManager.General import *


def normalize_features(x, max_magnitude=300):

    for ind in range(len(x)):
        for freq in range(len(x[ind])):
            for time in range(len(x[ind][freq])):
                for feature in range(len(x[ind][freq][time])):
                    if(x[ind][freq][time][feature] <= max_magnitude):
                        x[ind][freq][time][feature] /= max_magnitude
                    else:
                        x[ind][freq][time][feature] = 1.0


def create_pairs(x, y, n_channels=16):
    pairs = []
    labels = []
    # interval in order to channels pair with same channels
    interval = int(len(x) / (len(x) / n_channels))
    for ind in range(0, len(x), interval):
        # condition in order for the for not to exceed length
        if (int(len(x[ind:]) / n_channels) > 1):
            # inner interval as the individual increases
            inner_interval = int(len(x[ind + n_channels:]) / (len(x[ind + n_channels:]) / n_channels))
            for other_ind in range(ind + n_channels, len(x), inner_interval):
                for channel in range(n_channels):
                    pairs += [[x[ind + channel], x[other_ind + channel]]]
                    if (y[ind + channel] == y[other_ind + channel]):
                        labels += [1]
                    else:
                        labels += [0]

    pairs = np.array(pairs)
    labels = np.array(labels)

    return pairs, labels


def create_base_network(input_shape, kernel_size=(6, 6), final_dimension=12, regularization=0.011):
    ##model building
    model = Sequential()
    # convolutional layer with rectified linear unit activation
    # flatten since too many dimensions, we only want a classification output
    model.add(Conv2D(1, kernel_size=kernel_size,
                     activation='relu',
                     input_shape=input_shape, kernel_initializer=init,
                     bias_initializer='zeros',
                     kernel_regularizer=regularizers.l1(regularization),  # 0.011
                     bias_regularizer=regularizers.l1(regularization)))  # 0.011

    model.add(Dropout(0.5))
    model.add(Conv2D(1, kernel_size=kernel_size,
                     activation='relu', kernel_initializer=init,
                     bias_initializer='zeros',
                     kernel_regularizer=regularizers.l1(regularization),  # 0.011
                     bias_regularizer=regularizers.l1(regularization)))  # 0.011

    model.add(Dropout(0.5))
    # things to test in order to increase the performance of the mdel
    # play a little with the kernel sizes - test values: (6,6)
    # change the optimization function -
    model.add(Flatten())
    # embedding sizes with better results seem to be between [8,15[
    model.add(Dense(final_dimension, activation='softmax', kernel_initializer=init,
                    bias_initializer='zeros'))  # 13
    print(model.summary())
    return model


def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def cosine_distance(vects):
    x,y = vects
    #how should the normalization be done??
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)

    a = K.batch_dot(x, y, axes=1)

    b = K.batch_dot(x, x, axes=1)
    c = K.batch_dot(y, y, axes=1)

    return 1 - (a / (K.sqrt(b) * K.sqrt(c)))
    #line below is correct
    #return K.mean(1-K.abs(K.batch_dot(x, y, axes=1)))


class contrastive_loss():
    def __init__(self, margin):
        self.margin = margin

    def loss(self, y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


class Siamese:
    def __init__(self, input_shape, regularization=0.011, kernel_size=(6,6), final_dimension=12, learning_rate=0.0004, margin=1.2):
        self.base_network = create_base_network(input_shape, kernel_size, final_dimension, regularization)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = self.base_network(input_a)
        processed_b = self.base_network(input_b)

        distance = Lambda(cosine_distance,  # compare this results with euclidean
                          output_shape=cos_dist_output_shape)([processed_a,
                          processed_b])

        model = Model([input_a, input_b], distance)

        adam = Adam(lr=learning_rate)
        loss_function = contrastive_loss(margin)
        model.compile(loss=loss_function.loss, optimizer=adam, metrics=[accuracy])
        self.model = model

    def save_base_network(self, k):
        self.base_network.save_weights("./validation_models/base_network_partition_" + str(k) + ".h5")


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='loss', value=0.1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current < self.value:
            self.model.stop_training = True


# labels_path = "C:\\Users\\jorge\\DatasetsTFM\\KaggleData\\train.csv"
# train_path = "C:\\Users\\jorge\\DatasetsTFM\\KaggleData\\train"

labels_path = "data/train.csv"
train_path = "data/train"
init = keras.initializers.glorot_uniform(seed=0)


audiofiles = [os.path.join(train_path, f) for f in listdir(train_path) if isfile(join(train_path, f))]

limitator = None

labels_dict = get_labels(labels_path)

kwargs = {'p': .7, 'cut': True}

# Get a random permutation to remove ordering bias
np.random.shuffle(audiofiles)

# convert the list of files to numpy arrays and select a random smaller version if limitador is not None
X_path = np.array(audiofiles)[:limitator]

# Start the process of data extraction, spectrogram transformation and data enhancement
print('Generating train and test split')
X_train_path, X_test_path = train_test_split(X_path, test_size=0.3)

print('Getting test spectrograms')
X_test, Y_test = get_spects(X_test_path, labels_dict, **kwargs)

# print('Getting test spectrograms')
# X_train, Y_train = get_spects(X_train_path, labels_dict)

print('Getting train spectrograms + enhancement')
X_train, Y_train = get_spects_enhanced(X_train_path, labels_dict, **kwargs)

print('Getting even more data adding noise to whale calls')
X_enhanced, Y_enhanced = enhance_with_noise(X_train, Y_train)

X_train, Y_train = np.concatenate([X_train, X_enhanced]), np.concatenate([Y_train, Y_enhanced])

print('Test', X_test.shape)

print('Train', X_train.shape)


time = len(X_train[0])
freq = len(X_train[0][0])
X_train = X_train.reshape(X_train.shape[0], time, freq, 1)
X_test = X_test.reshape(X_test.shape[0], time, freq, 1)
input_shape = (time, freq, 1)
#more reshaping
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

normalization_factor = 1.04842448e+02
epochs = 10
learning_rate = 1.49586563e-03

margin_factor = 1.18389688e+00
regularization = 1.65531895e-02
final_dimension = 10
kernel_size = (5, 11)

normalize_features(X_train, max_magnitude=normalization_factor)
normalize_features(X_test, max_magnitude=normalization_factor)

tr_pairs, tr_y = create_pairs(X_train, Y_train)

s = Siamese(input_shape,
            learning_rate=learning_rate,
            final_dimension=final_dimension,
            kernel_size=kernel_size,
            regularization=regularization,
            margin=margin_factor)



history = s.model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, epochs=epochs)

y_pred = s.model.predict([tr_pairs[:, 0],
                        tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
