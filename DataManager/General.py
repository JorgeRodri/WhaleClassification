import numpy as np
import csv


def enhance_with_noise(X, Y, noise=0.28):
    whale_index, = np.where(Y == '1')
    no_whale_index, = np.where(Y == '0')
    x_enhanced = []
    y_enhanced = []
    for s_i in whale_index:
        new_x = X[s_i] + noise*X[np.random.choice(no_whale_index)]
        x_enhanced.append(new_x)
        y_enhanced.append(1)
    return np.array(x_enhanced),  np.array(y_enhanced)


def get_labels(labels_path):
    labels = dict()
    with open(labels_path, 'r') as f:
        reader = csv.reader(f, dialect='excel')
        for row in reader:
            labels[row[0]] = row[1]
    return labels


def normalize_max(x):
    return x / x.max()


def normalize_log(x, c=1):
    return np.log(x + c)


def normalize_box_cox(x, a, b=1):
    """
    Box plot transformation for the spectrograns, with b=0 we get a one Parameter Box-Cox but since spectrograms 0
    values it is preferable to have a vlue that cannot cause problems such as 1
    :param x: np.array with the data of the spectrograms
    :param a: first box plot parameter, the one that is most interesting to analyze
    :param b: a normalization value to ensure logarithms value don't grow too much
    :return: normalized numpy array
    """

    if a == 0:
        return np.log(x + b)

    else:
        return ((x + b)**a - 1) / a
