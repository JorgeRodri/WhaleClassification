from scipy.io import wavfile
from scipy import signal
import numpy as np


# sample_rate, audio = wavfile.read(path_to_wav_file)
def get_spec_par_blackman(file):
#     save_name='../KaggleData/spec_blackman/'

    with aifc.open(file, 'r') as f:
        nframes = f.getnframes()
        strsig = f.readframes(nframes)
        data = numpy.fromstring(strsig, numpy.short).byteswap()
    nfft = 256  # Length of the windowing segments
    fs = 2
    pxx, freqs, bins, im = plt.specgram(data, NFFT=nfft, Fs=fs, window=numpy.blackman(256))
    plt.axis('off')
#     plt.savefig(save_name + file[20:-5] + '.png',
#                 dpi=100,  # Dots per inch
#                 frameon='false',
#                 aspect='normal',
#                 bbox_inches='tight',
#                 pad_inches=0)  # Spectrogram saved as a .png
    return pxx, freqs, bins, im

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


f = aifc.open(file+audio_file, 'r')
nframes = f.getnframes()
strsig = f.readframes(nframes)
data = numpy.fromstring(strsig, numpy.short).byteswap()
# f.close()
sample_freq, segment_time, spec_data = log_specgram(data, f.getframerate())
sample_freq, segment_time, spec_data = signal.spectrogram(data, f.getframerate())
plt.pcolormesh(segment_time, sample_freq, spec_data )
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()