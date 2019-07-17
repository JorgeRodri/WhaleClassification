import os
import time
import datetime

import numpy as np
from os.path import isfile, join
from sklearn.model_selection import train_test_split

from DataManager.Audio import get_spects, get_spects_enhanced
from DataManager.General import get_labels, enhance_with_noise

t1 = time.time()
save_path = "/home/jorge/PycharmProjects/AudioExtraction/result_graphs"
numpy_save_path = "/home/jorge/PycharmProjects/AudioExtraction/numpy_data"
labels_path = "/home/jorge/Documents/DatasetsTFM/KaggleData/train.csv"
train_path = "/home/jorge/Documents/DatasetsTFM/KaggleData/train"
tag = 'prueba_100'


if __name__ == '__main__':
    np.random.seed(21052711)

    print('Reading paths to audiofiles, started at {}'.format(datetime.datetime.now()))
    audiofiles = [os.path.join(train_path, f) for f in os.listdir(train_path) if isfile(join(train_path, f))]

    labels_dict = get_labels(labels_path)

    label_by_order = np.vectorize(lambda path: labels_dict[path.split('\\')[-1]])
    np.random.shuffle(audiofiles)
    X_path = np.array(audiofiles)  # limitador
    # Y = label_by_order(audiofiles)

    print('Generating train and test split')
    X_train_path, X_test_path = train_test_split(X_path, test_size=0.3)

    print('Getting test spectrograms')
    X_test, Y_test = get_spects(X_test_path, labels_dict, cut=False)

    print('Getting train spectrograms + enhancement')
    X_train, Y_train = get_spects_enhanced(X_train_path, labels_dict, cut=False)

    print('Getting even more data adding noise to whale calls')
    X_enhanced, Y_enhanced = enhance_with_noise(X_train, Y_train)
    X_train, Y_train = np.concatenate([X_train, X_enhanced]), np.concatenate([Y_train, Y_enhanced])

    t2 = time.time()

    np.save(os.path.join(numpy_save_path, 'xtrain_no_cut'), X_train)
    np.save(os.path.join(numpy_save_path, 'xtest_no_cut'), X_test)
    np.save(os.path.join(numpy_save_path, 'ytrain_no_cut'), Y_train)
    np.save(os.path.join(numpy_save_path, 'ytest_no_cut'), Y_test)

    print('needed time: {}'.format(t2-t1))
    print('Finished at {}'.format(datetime.datetime.now()))
