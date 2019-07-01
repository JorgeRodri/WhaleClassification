import aifc
import numpy
import matplotlib.pyplot as plt

file = 'C:/Users/jorge/Downloads/small_data_sample/{}right_whale/'
audio_file = 'train{}.aiff'

file_number = 5
if file_number<=5:
    file = file.format('no_')
else:
    file = file.format('')
audio_file = 'train{}.aiff'.format(file_number)

f = aifc.open(file+audio_file, 'r')
nframes = f.getnframes()
strsig = f.readframes(nframes)
data = numpy.fromstring(strsig, numpy.short).byteswap()
# f.close()

nfft = 256  # Length of the windowing segments
fs = 2
pxx, freqs, bins, im = plt.specgram(data, NFFT=nfft, Fs=fs, window=numpy.blackman(256))
plt.axis('off')

# OPTION 1
import librosa
import librosa.display

y, sr = librosa.load(file + audio_file, duration=196)
ps = librosa.feature.melspectrogram(y=y, sr=sr, fmax = 1024)
# ps = librosa.core.stft(y=y)
print(ps.shape)
librosa.display.specshow(ps, y_axis='mel', x_axis='time')
plt.ylim()

# OPTION 2
import librosa
import librosa.display

y, sr = librosa.load(file + audio_file, duration=196)
ps = librosa.feature.melspectrogram(y=y, sr=sr, fmax = 1024)
# ps = librosa.core.stft(y=y)
print(ps.shape)
librosa.display.specshow(ps, y_axis='mel', x_axis='time')
plt.ylim()
