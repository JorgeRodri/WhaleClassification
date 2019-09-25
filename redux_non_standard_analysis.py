import os
import time
import datetime
from math import floor

import numpy as np
from os.path import isfile, join
import aifc

from DataManager.Audio import read_aiff, get_spects
from DataManager.General import get_labels, enhance_with_noise

import pandas as pd
from os import listdir

import matplotlib.pyplot as plt


def fill_array(data):
    if data.shape[0] < 4000:
        corrected_data = np.zeros(4000)
        corrected_data[:data.shape[0]] = data
    elif data.shape[0] > 4000:
        corrected_data = data[floor((data.shape[0] - 4000)/2):
                              data.shape[0] - floor((data.shape[0]-4000)/2)]
    else:
        corrected_data = data
    return corrected_data


def read_aiff_detailed(file):
    s = aifc.open(file, 'r')
    framerate = s.getframerate()
    nframes = s.getnframes()
    strsig = s.readframes(nframes)
    data = np.fromstring(strsig, np.short).byteswap()
    # corrected_data = fill_array(data)
    s.close()
    return [data.shape, framerate, nframes]


train_redux_path = "data/train2/"

reduxfiles = [os.path.join(train_redux_path, f) for f in listdir(train_redux_path) if isfile(join(train_redux_path, f))]

no_std_files = [[i,
                 i.split('/')[-1][-5],
                 read_aiff_detailed(i)]
                for i in reduxfiles]

df_non_std = pd.DataFrame(no_std_files, columns=['file', 'label', 'audio_samples', 'framerate', 'nframes'])

print(df_non_std['label'].value_counts())
print(df_non_std[df_non_std['audio_samples'] != 4000]['label'].value_counts().head())
print(df_non_std[df_non_std['audio_samples'] != 4000]['framerate'].value_counts().head())
print(df_non_std[df_non_std['audio_samples'] != 4000]['nframes'].value_counts().head())

# TODO: find if whales calls are centered
