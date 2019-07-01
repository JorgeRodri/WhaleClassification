import tensorflow as tf
import aifc
import os
import numpy as np
from matplotlib import mlab
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split

import datetime
import csv
import matplotlib.pyplot as plt


def get_labels(labels_path):
    labels = dict()
    with open(labels_path, 'r') as f:
        reader = csv.reader(f, dialect='excel')
        for row in reader:
            labels[row[0]] = row[1]
    return labels


def ReadAIFF(file):
    s = aifc.open(file,'r')
    nFrames = s.getnframes()
    strSig = s.readframes(nFrames)
    return np.fromstring(strSig, np.short).byteswap()


def get_spects(onlyfiles, labels, p=0.7, cut=True):
    if cut:
        top_hz = 40
    else:
        top_hz = -1
    sps = []
    y = []
    for file_path in onlyfiles:
        s = ReadAIFF(file_path)
        s = s[int(s.shape[0] * (1 - p) / 2): int(s.shape[0] * (1 + p) / 2)]
        y.append(labels[file_path.split("\\")[-1]])
        params = {'NFFT': 256, 'Fs': 2000, 'noverlap': 192}
        P, freqs, bins = mlab.specgram(s, **params)
        sps.append(P[:top_hz])
    return np.array(sps), np.array(y)


def get_spects_enhanced(onlyfiles, labels, p=0.7, cut=True):
    if cut:
        top_hz = 40
    else:
        top_hz = -1
    sps = []
    y = []
    for file_path in onlyfiles:
        s = ReadAIFF(file_path)
        this_label = labels[file_path.split("\\")[-1]]
        s1 = s[:int(s.shape[0] * p)]
        y.append(this_label)
        s2 = s[int(s.shape[0] * (1-p)/2): int(s.shape[0] * (1+p)/2)]
        y.append(this_label)
        s3 = s[int(s.shape[0] * (1-p)):]
        y.append(this_label)
        params = {'NFFT': 256, 'Fs': 2000, 'noverlap': 192}
        P1, freqs, bins = mlab.specgram(s1, **params)
        sps.append(P1[:top_hz, :])
        P2, freqs, bins = mlab.specgram(s2, **params)
        sps.append(P2[:top_hz, :])
        P3, freqs, bins = mlab.specgram(s3, **params)
        sps.append(P3[:top_hz, :])
    return np.array(sps), np.array(y)


def enhance_with_noise(X, Y):
    whale_index, = np.where(Y == '1')
    no_whale_index, = np.where(Y == '0')
    x_enhanced = []
    y_enhanced = []
    for s_i in whale_index:
        new_x = X[s_i] + 0.28*X[np.random.choice(no_whale_index)]
        x_enhanced.append(new_x)
        y_enhanced.append(1)
    return np.array(x_enhanced),  np.array(y_enhanced)


if __name__ == '__main__':

    save_path = "C:\\Users\\jorge\\PycharmProjects\\AudioExtraction\\result_graphs"
    labels_path = "C:\\Users\\jorge\\DatasetsTFM\\KaggleData\\train.csv"
    train_path = "C:\\Users\\jorge\\DatasetsTFM\\KaggleData\\train"

    tag = 'prueba_100'

    print('Reading paths to audiofiles, started at {}'.format(datetime.datetime.now()))
    audiofiles = [os.path.join(train_path, f) for f in listdir(train_path) if isfile(join(train_path, f))]

    labels_dict = get_labels(labels_path)

    label_by_order = np.vectorize(lambda x: labels_dict[x.split('\\')[-1]])
    np.random.shuffle(audiofiles)
    X_path = np.array(audiofiles)[:1000]  # limitador
    # Y = label_by_order(audiofiles)

    print('Generating train and test split')
    X_train_path, X_test_path = train_test_split(X_path, test_size=0.3)

    print('Getting test spectrograms')
    X_test, Y_test = get_spects(X_test_path, labels_dict)

    print('Getting train spectrograms + enhancement')
    X_train, Y_train = get_spects_enhanced(X_train_path, labels_dict)

    print('Getting even more data adding noise to whale calls')
    X_enhanced, Y_enhanced = enhance_with_noise(X_train, Y_train)
    X_train, Y_train = np.concatenate([X_train, X_enhanced]), np.concatenate([Y_train, Y_enhanced])

    print('Setting parameters and data reshape for the training')
    opt = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    Y_train, Y_test = Y_train.astype(int), Y_test.astype(int)
    Y_train, Y_test = tf.keras.utils.to_categorical(Y_train, 2), tf.keras.utils.to_categorical(Y_test, 2)

    # Normalization of the data
    X_norm = np.linalg.norm(X_train)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1) / X_norm
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1) / X_norm

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(20, kernel_size=(7, 7), activation=tf.nn.relu, input_shape=X_train.shape[1:]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(40, kernel_size=(7, 7), activation=tf.nn.relu, name='Conv2'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu, name="Dense1"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax, name="Softmax")
        ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print('Training model')
    history = model.fit(X_train, Y_train, epochs=10, verbose=1)
    score = model.evaluate(X_test, Y_test)

    print(score)
    print('Finished at {}, saving the results as graphs.'.format(datetime.datetime.now()))

    #Accuracy plot
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')

    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(save_path + tag + 'model_accuracy' + str(score[1]) + '.pdf')
    plt.close()
    #Loss plot
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(save_path + tag + 'model_loss' + str(score[1]) + '.pdf')

    #
    # from sklearn.metrics import classification_report,confusion_matrix
    # import numpy as np
    # #Compute probabilities
    # y_train = data_generator.classes
    # Y_pred = nn.predict_generator(data_generator, steps=80)
    # # print(Y_pred.shape)
    # #Assign most probable label
    # y_pred = np.argmax(Y_pred, axis=1)
    # #Plot statistics
    # print('Analysis of results')
    # target_names = ['no_whale', 'whale']
    # print(classification_report(y_train, y_pred, target_names=target_names))
    # print(confusion_matrix(y_train, y_pred))

    # #Confusion Matrix
    # from sklearn.metrics import classification_report,confusion_matrix
    # import numpy as np
    # #Compute probabilities
    # y_test = test_gen.classes
    # Y_pred = nn.predict_generator(test_gen, steps=20)
    # # print(Y_pred.shape)
    # #Assign most probable label
    # y_pred = np.argmax(Y_pred, axis=1)
    # #Plot statistics
    # print('Analysis of results')
    # target_names = ['no_whale', 'whale']
    # print(classification_report(y_test, y_pred,target_names=target_names))
    # print(confusion_matrix(y_test, y_pred))
