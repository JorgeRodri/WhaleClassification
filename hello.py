import numpy as np
import os


def get_wavs(files_path):
    waves = []
    directories = os.listdir(path)
    for folder in directories:
        try:
            file_list = os.listdir(os.path.join(path, folder))
            audios = [os.path.join(path, f) for f in file_list if f[-4:] == '.wav']
            waves += audios
        except NotADirectoryError:
            print(folder)
    return waves


path = 'C:/Users/jorge/DatasetsTFM/2015DCLDEWorkshop/SocalLFDevelopmentData'
if __name__ == '__main__':
    files = get_wavs(path)
    for file in files:
        pass