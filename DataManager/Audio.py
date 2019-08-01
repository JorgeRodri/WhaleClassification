import aifc
import numpy as np
from matplotlib import mlab


def read_aiff(file):
    s = aifc.open(file, 'r')
    nframes = s.getnframes()
    strsig = s.readframes(nframes)
    return np.fromstring(strsig, np.short).byteswap()


def get_spects(onlyfiles, labels=N, p=0.7, cut=True):
    if cut:
        top_hz = 40
    else:
        top_hz = -1
    sps = []
    y = []
    for file_path in onlyfiles:
        s = read_aiff(file_path)
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
        s = read_aiff(file_path)
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
