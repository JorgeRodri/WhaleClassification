import os
import time
import datetime

import numpy as np
from os.path import isfile, join

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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
limitador_original = None  # None for complete data
limitador_redux = 0  # None for complete data

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
    X_path = np.array(audiofiles)[:limitador_original]
    X_redux_path = np.array(reduxfiles)[:limitador_redux]

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

    print('Getting even more data adding noise to whale calls')
    X_enhanced, Y_enhanced = enhance_with_noise(X_train, Y_train)

    print(X_train.shape, X_enhanced.shape, Y_train.shape, Y_enhanced.shape)

    X_train, Y_train = np.concatenate([X_train, X_enhanced]), np.concatenate([Y_train, Y_enhanced])

    print('Test', X_test.shape)

    print('Train', X_train.shape)


    def normalize(x):
        return x / x.max()


    X_train = np.array(list(map(normalize, X_train)))
    X_test = np.array(list(map(normalize, X_test)))

    print('Test: Whale %.4f, Not Whale %.4f' % ((Y_test == '1').sum() / Y_test.shape[0],
                                                (Y_test == '0').sum() / Y_test.shape[0]))

    print('Train: Whale %.4f, Not Whale %.4f' % ((Y_train == '1').sum() / Y_train.shape[0],
                                                 (Y_train == '0').sum() / Y_train.shape[0]))

    print('Tamaño en memoria de los datos de training aprox: %.2fGB' % (X_train.nbytes / 2 ** 10 / 2 ** 10 / 2 ** 10))

    # Prepare data for neural network
    Y_train, Y_test = Y_train.astype(int), Y_test.astype(int)
    Y_train, Y_test = tf.keras.utils.to_categorical(Y_train, 2), tf.keras.utils.to_categorical(Y_test, 2)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    opt = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(20, kernel_size=(7, 7), activation=tf.nn.relu,
                               input_shape=X_train.shape[1:], name='Conv1'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(40, kernel_size=(7, 7), activation=tf.nn.relu, name='Conv2'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu, name="Dense1"),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax, name="Softmax")
    ])

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print('Training model')
    print(model.summary())
    history = model.fit(X_train, Y_train, epochs=20, verbose=1)
    score = model.evaluate(X_test, Y_test)

    print(score)
    print('Finished at {}, saving the results as graphs.'.format(datetime.datetime.now()))

    # Accuracy plot
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')

    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    # plt.savefig(save_path + tag + 'model_accuracy' + str(score[1]) + '.pdf')
    plt.show()

    # Loss plot
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    # plt.savefig(save_path + tag + 'model_loss' + str(score[1]) + '.pdf')
    plt.show()

    # Confusion Matrix

    # Compute probabilities
    Y_pred = model.predict(X_test)
    # Assign most probable label
    y = np.argmax(Y_test, axis=1)
    y_hat = np.argmax(Y_pred, axis=1)
    # Plot statistics
    print('Analysis of results')
    target_names = ['no_whale', 'whale']
    print(classification_report(y, y_hat, target_names=target_names))
    print(confusion_matrix(y, y_hat))
