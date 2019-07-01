from get_Spect_train_data import *
import time
import datetime


def get_spects(onlyfiles, labels, p=0.7, cut=True):
    if cut:
        top_hz = 40
    else:
        top_hz = -1
    sps = []
    y = []
    for file_path in onlyfiles:
        s = ReadAIFF(file_path)
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
        s = ReadAIFF(file_path)
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


def enhance_with_noise(X, Y):
    whale_index, = np.where(Y == '1')
    no_whale_index, = np.where(Y == '0')
    x_enhanced = []
    y_enhanced = []
    for s_i in whale_index:
        new_x = X[s_i] + 0.28*X[np.random.choice(no_whale_index)]
        x_enhanced.append(new_x)
        y_enhanced.append(1)
    return np.array(x_enhanced),  np.array(y_enhanced)


if __name__ == '__main__':
    np.random.seed(21052711)
    t1 = time.time()
    save_path = "/home/jorge/PycharmProjects/AudioExtraction/result_graphs"
    numpy_save_path = "/home/jorge/PycharmProjects/AudioExtraction/numpy_data"
    train_path = "/home/jorge/Documents/DatasetsTFM/KaggleDataRedux/train2"
    tag = 'prueba_100'

    print('Reading paths to audiofiles, started at {}'.format(datetime.datetime.now()))
    audiofiles = [os.path.join(train_path, f) for f in listdir(train_path) if isfile(join(train_path, f))]

    # TODO: [f[-5] for f in audiofiles][:10]

    np.random.shuffle(audiofiles)
    X_path = np.array(audiofiles)  # limitador

    print('Generating train and test split')
    X_train_path, X_test_path = train_test_split(X_path, test_size=0.3)

    print('Getting test spectrograms')
    X_test, Y_test = get_spects(X_test_path, labels_dict, cut=False)

    print('Getting train spectrograms + enhancement')
    X_train, Y_train = get_spects_enhanced(X_train_path, labels_dict, cut=False)

    print('Getting even more data adding noise to whale calls')
    X_enhanced, Y_enhanced = enhance_with_noise(X_train, Y_train)
    X_train, Y_train = np.concatenate([X_train, X_enhanced]), np.concatenate([Y_train, Y_enhanced])

    t2 = time.time()

    np.save(os.path.join(numpy_save_path, 'xtrain_no_cut'), X_train)
    np.save(os.path.join(numpy_save_path, 'xtest_no_cut'), X_test)
    np.save(os.path.join(numpy_save_path, 'ytrain_no_cut'), Y_train)
    np.save(os.path.join(numpy_save_path, 'ytest_no_cut'), Y_test)

    print('needed time: {}'.format(t2-t1))
    print('Finished at {}'.format(datetime.datetime.now()))
