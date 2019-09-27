import aifc
import numpy as np
from matplotlib import mlab


def read_aiff(file):
    with aifc.open(file, 'r') as s:
        nframes = s.getnframes()
        strsig = s.readframes(nframes)
    return np.fromstring(strsig, np.short).byteswap()


def get_spects(onlyfiles, labels, p=0.85, top_hz=-1):

    sps = []
    y = []
    for file_path in onlyfiles:
        s = read_aiff(file_path)
        if s.shape[0] != 4000:
            continue
        s = s[int(s.shape[0] * (1 - p) / 2): int(s.shape[0] * (1 + p) / 2)]
        # check if it comes from redux data it is not in the labels dictionary
        try:
            this_y = labels[file_path.split("/")[-1]]
        except KeyError:
            this_y = file_path[-5]
        except Exception as e:
            print('Unknown error: ', e)
            this_y = '.'
        finally:
            y.append(this_y)
        params = {'NFFT': 256, 'Fs': 2000, 'noverlap': 192}
        sp, freqs, bins = mlab.specgram(s, **params)
        sps.append(sp[:top_hz])
    return np.array(sps), np.array(y)


def get_spects_enhanced(onlyfiles, labels, p=0.85, top_hz=-1):
    """

    :param onlyfiles:
    :param labels:
    :param p:
    :param top_hz: basic value should be 40 or -1 to keep the spectrpgram whole
    :return:
    """
    sps = []
    y = []
    for file_path in onlyfiles:
        s = read_aiff(file_path)
        if s.shape[0] != 4000:
            continue
        # check if it comes from redux data it is not in the labels dictionary
        try:
            this_label = labels[file_path.split("/")[-1]]
        except KeyError:
            this_label = file_path[-5]
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


def get_spects_enhance_only_whales(onlyfiles, labels, p=0.85, top_hz=-1):

    sps = []
    y = []
    for file_path in onlyfiles:
        s = read_aiff(file_path)
        if s.shape[0] != 4000:
            continue
        # check if it comes from redux data it is not in the labels dictionary
        try:
            this_label = labels[file_path.split("/")[-1]]
        except KeyError:
            this_label = file_path[-5]

        if this_label == '1':
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

        else:
            s = s[int(s.shape[0] * (1-p)/2): int(s.shape[0] * (1+p)/2)]
            y.append(this_label)
            params = {'NFFT': 256, 'Fs': 2000, 'noverlap': 192}
            p, freqs, bins = mlab.specgram(s, **params)
            sps.append(p[:top_hz, :])
    return np.array(sps), np.array(y)
