import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import aifc
from os import walk
import numpy
import os
import librosa
import librosa.display
import scipy.signal as signal


def get_spec_mel(file):
    save_name = '../Spectrograms/mel/'
    plt.axis('off')
    print(save_name, file[20:-5],'.png')
    plt.savefig(save_name + file[20:-5] + '.png',
                dpi=100,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png
    plt.close()


def get_spec_scipy(file):
    save_name = '../Spectrograms/scipy/'
    with aifc.open(file, 'r') as f:
        nframes = f.getnframes()
        strsig = f.readframes(nframes)
        data = numpy.fromstring(strsig, numpy.short).byteswap()
        f, t, sxx = signal.spectrogram(data)
        plt.pcolormesh(t, f, sxx)
    plt.axis('off')
    plt.savefig(save_name + file[20:-5] + '.png',
                dpi=100,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png
    plt.close()


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]


def get_files(folder_path):
    audio_files = []
    for (dirpath, dirnames, filenames) in walk(folder_path):
        audio_files.extend(filenames)
        return audio_files


path = '../train/'


files = get_files(path)
split = split_list(files, 8)

part = split[0]


for file in part:
    get_spec_mel(path+file)

