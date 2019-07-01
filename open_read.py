import os
import csv
import matplotlib.pyplot as plt
import librosa
from scipy import signal
import datetime
from scipy.io import wavfile
import pandas as pd
from math import floor


def extract_audio(audio, ini, fin):
    # TODO recortes
    pass


def get_labels_df(path):
    list_labels = os.listdir(path)
    df = pd.DataFrame(columns=['Location', 'Opt', 'Whale', 'Start', 'End', 'Call'])

    for file in list_labels:
        if file[-4:] == '.csv':
            labels_df = pd.read_csv(os.path.join(path, file), header=None,
                                 names=['Location', 'Opt', 'Whale', 'Start', 'End', 'Call'])
            df = df.append(labels_df)
    return df


def parse_file_name(file_name, how='list'):
    types = ['CINMS', 'DCPP']
    if file_name[0:5] == types[0]:
        first = file_name[0:5]
        extra = file_name[5:8]
        second = file_name[9:12]
        third = file_name[13:19]
        forth = file_name[20:26]
    elif file_name[0:4] == types[1]:
        first = file_name[0:4]
        extra = file_name[4:7]
        second = file_name[8:11]
        third = file_name[12:18]
        forth = file_name[19:25]
    else:
        raise Exception
    if how == 'str':
        return '%s %s 20%s-%s-%s %s:%s:%s' % (first, second, third[0:2], third[2:4], third[4:6],
                                              forth[0:2], forth[2:4], forth[4:6])
    elif how == 'list':
        return first, second, third[0:2], third[2:4], third[4:6], forth[0:2], forth[2:4], forth[4:6]
    else:
        loc = first
        opt = extra
        date_time = datetime.datetime.strptime('20%s-%s-%s %s:%s:%s' % (third[0:2], third[2:4], third[4:6],
                                                                        forth[0:2], forth[2:4], forth[4:6]),
                                               '%Y-%m-%d %H:%M:%S')
        return loc, opt, date_time


def parse_time2(file_name):
    first, second, third, forth = file_name.split('_')
    return '%s %s 20%s-%s-%s %s:%s:%s' % (first, second, third[0:2], third[2:4], third[4:6],
                                          forth[0:2], forth[2:4], forth[4:6])


def get_wavs(files_path):
    waves = []
    directories = os.listdir(files_path)
    for folder in directories:
        try:
            file_list = os.listdir(os.path.join(files_path, folder))
            audios = [os.path.join(os.path.join(files_path, folder), f) for f in file_list if f[-4:] == '.wav']
            waves += audios
        except NotADirectoryError:
            continue
    return waves


if __name__ == '__main__':
    dir_path = 'C:\\Users\\jorge\\DatasetsTFM\\2015DCLDEWorkshop\\AnalystAnnotations\\SocalLFDevelopmentData'
    labels = get_labels_df(dir_path)
    labels['Start'] = pd.to_datetime(labels['Start'], 'raise')
    labels['End'] = pd.to_datetime(labels['End'], 'raise')

    # from test import *

    path = 'C:\\Users\\jorge\\DatasetsTFM\\2015DCLDEWorkshop\\SocalLFDevelopmentData'
    audio_list = get_wavs(path)

    for i in audio_list:
        print(parse_file_name(i.split('\\')[-1], 'other'))
        location, identifier, date = parse_file_name(i.split('\\')[-1], 'other')
        fs, data = wavfile.read(i)
        difference = datetime.timedelta(seconds=data.shape[0]/fs)
        this_labels = labels[(labels['Start'] > date) &
                             (labels['End'] < date + difference) &
                             (labels['Location'] == location) &
                             (labels['Opt'] == identifier[-1])]
        c = 0
        for index, row in this_labels.iterrows():
            start = row['Start'] - date
            finish = row['End'] - date
            s = 0.5
            clip = data[floor(start.seconds*fs - s*fs):floor(finish.seconds*fs + s*fs + 1)]
            plt.plot(clip)
            plt.show()

            f, t, sxx = signal.spectrogram(clip)
            plt.title(row['Call'])
            plt.pcolormesh(t, f, sxx)
            plt.show()
            # librosa.display.specshow(ps, y_axis='mel', x_axis='time')
            c += 1
            if c > 5:
                break
        break
    wav = audio_list[0]
    fs, data = wavfile.read(wav)
    print(labels.head())
    print(labels['Location'].unique(), labels['Opt'].unique(), labels['Whale'].unique(), labels['Call'].unique())

    print('%dh, %dm, %ds' % (data.shape[0]/fs/60//60, data.shape[0]/fs/60 % 60, data.shape[0]/2000 % 60))

    '''
    Parece que lo mejor seria hacerlo audio a audio, se cargan las labels, se abre cada audio y se extraen de el
    for audio en audio_list
        label_buenas <- labels[fecha inicio del audio-fechafinal, loc correcta]
        for label in labels_buenas
            cut audio[label[init]:label[fibal]
    '''
