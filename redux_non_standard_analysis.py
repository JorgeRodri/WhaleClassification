import os
import time
import datetime

import numpy as np
from os.path import isfile, join
from sklearn.model_selection import train_test_split

from DataManager.Audio import read_aiff, get_spects
from DataManager.General import get_labels, enhance_with_noise

import tensorflow as tf
from os import listdir

import matplotlib.pyplot as plt


train_redux_path = "data/train2/"

reduxfiles = [os.path.join(train_redux_path, f) for f in listdir(train_redux_path) if isfile(join(train_redux_path, f))]

print([(i.split('/')[-1][-5], read_aiff(i).shape[0]) for i in reduxfiles if read_aiff(i).shape[0] != 4000])

