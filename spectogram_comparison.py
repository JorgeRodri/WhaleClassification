import aifc
import numpy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.signal as signal
import librosa
import librosa.display
import os
import time


def get_mlab_spec(data, nfft=256, fs=2000, noverlap=192, window=mlab.window_hanning):
    P, freqs, bins = mlab.specgram(data, NFFT=nfft, Fs=fs, noverlap=noverlap if noverlap < nfft else nfft//2, window=window)
    plt.pcolormesh(bins, freqs, P)
    return plt


def get_plt_spec(data, nfft=256, fs=2, window=mlab.window_hanning):
    pxx, freqs, bins, im = plt.specgram(data, NFFT=nfft, Fs=fs, noverlap=nfft//2, window=window)
    # plt.axis('off')
    return plt


def get_scipy_spec(data, nfft=256, fs=256, window='hann'):
    f, t, sxx = signal.spectrogram(data, nfft=nfft, fs=fs, noverlap=nfft//2, window=window)
    plt.pcolormesh(t, f, sxx)
    # plt.axis('off')
    return plt


def get_librosa_spec(file_path, duration, nfft=256, fs=2, window='hann'):
    if fs:
        pass
    y, sr = librosa.load(file_path, duration=duration)
    ps = librosa.core.stft(y=y, n_fft=nfft, window=window)
    librosa.display.specshow(ps, y_axis='mel', x_axis='time')
    # plt.axis('off')
    plt.ylim(ymax=256)
    return plt


def get_audio_data(clip_path):
    with aifc.open(clip_path, 'r') as f:
        nframes = f.getnframes()
        strsig = f.readframes(nframes)
        framerate = f.getframerate()
        data = numpy.fromstring(strsig, numpy.short).byteswap()
    return data,  nframes, framerate


def get_any_spec(clip, spect_type, save_name, **params):
    data, nframes, framerate = get_audio_data(clip)
    if spect_type == 'plt':
        spectrogram = get_plt_spec(data, **params)
    elif spect_type == 'scipy':
        spectrogram = get_scipy_spec(data, **params)
    elif spect_type == 'librosa':
        spectrogram = get_librosa_spec(clip, nframes//framerate, **params)
    elif spect_type == 'mlab':
        spectrogram = get_mlab_spec(data, **params)
    else:
        raise Exception('Not a valid spectogram type')

    spectrogram.savefig(save_name,
                        dpi=100,  # Dots per inch
                        frameon='false',
                        aspect='normal',
                        bbox_inches='tight',
                        pad_inches=0)  # Spectrogram saved as a .png


if __name__ == '__main__':
    #winners
    # params = {'NFFT': 256, 'Fs': 2000, 'noverlap': 192}
    # s = aifc.open(file + audio_file, 'r')
    # nFrames = s.getnframes()
    # strSig = s.readframes(nFrames)
    # s = np.fromstring(strSig, np.short).byteswap()
    dirname = os.path.dirname(__file__)
    spect_types = ['plt', 'scipy', 'librosa', 'mlab']
    list_window = ['blackman']
    list_nfft = [256, 128]
    list_fs = [2, 4]
    save_name = 'spectrograms\\%s_%s_fs%d_nfft%d_%s.png'
    clip = 'test5.aiff'
    data_path = 'C:\\Users\\jorge\\DatasetsTFM\\KaggleData\\test\\'

    for i in spect_types:
        for j in list_fs:
            for k in list_nfft:
                for h in list_window:
                    time.sleep(5)
                    get_any_spec(os.path.join(data_path, clip), i,
                                 os.path.join(dirname, save_name % (clip, i, j, k, h)),
                                 nfft=k, fs=j, window=numpy.blackman(k))
                    plt.close()
