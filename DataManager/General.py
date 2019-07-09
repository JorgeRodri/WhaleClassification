import numpy as np
import csv


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


def get_labels(labels_path):
    labels = dict()
    with open(labels_path, 'r') as f:
        reader = csv.reader(f, dialect='excel')
        for row in reader:
            labels[row[0]] = row[1]
    return labels
