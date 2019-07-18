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

if __name__ == "__main__":
    print(train_path)
