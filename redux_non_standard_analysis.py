import os
import time
import datetime

import numpy as np
from os.path import isfile, join
import aifc

from DataManager.Audio import read_aiff, get_spects
from DataManager.General import get_labels, enhance_with_noise

import pandas as pd
from os import listdir

import matplotlib.pyplot as plt


def read_aiff_detailed(file):
    s = aifc.open(file, 'r')
    framerate = s.getframerate()
    nframes = s.getnframes()
    strsig = s.readframes(nframes)
    return {'data': np.fromstring(strsig, np.short).byteswap(),
            'framerate': framerate,
            'nframes': nframes}


train_redux_path = "data/train2/"

reduxfiles = [os.path.join(train_redux_path, f) for f in listdir(train_redux_path) if isfile(join(train_redux_path, f))]

no_std_files = [[i,
                 i.split('/')[-1][-5],
                 read_aiff_detailed(i)['data'].shape[0],
                 read_aiff_detailed(i)['framerate'],
                 read_aiff_detailed(i)['nframes']]
                for i in reduxfiles]

df_non_std = pd.DataFrame(no_std_files, columns=['file', 'label', 'audio_samples', 'framerate', 'nframes'])

print(df_non_std['label'].value_counts())
print(df_non_std[df_non_std['audio_samples'] != 4000]['label'].value_counts().head())
print(df_non_std[df_non_std['audio_samples'] != 4000]['framerate'].value_counts().head())
print(df_non_std[df_non_std['audio_samples'] != 4000]['nframes'].value_counts().head())
