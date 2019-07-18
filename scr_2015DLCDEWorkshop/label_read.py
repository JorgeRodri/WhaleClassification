import os
import csv
import pandas as pd


dir_path = 'C:\\Users\\jorge\\DatasetsTFM\\2015DCLDEWorkshop\\AnalystAnnotations\\SocalLFDevelopmentData'


def get_labels_df(path):
    list_labels = os.listdir(path)
    df = pd.DataFrame(columns=['Location', 'Opt', 'Whale', 'Start', 'End', 'Call'])

    for file in list_labels:
        if file[-4:] == '.csv':
            labels = pd.read_csv(os.path.join(path, file), header=None,
                                 names=['Location', 'Opt', 'Whale', 'Start', 'End', 'Call'])
            df = df.append(labels)
    return df


if __name__ == '__main__':

    labels = get_labels_df(dir_path)
    labels['Start'] = pd.to_datetime(labels['Start'], 'raise')
    labels['End'] = pd.to_datetime(labels['End'], 'raise')
    print(len(labels))
    print(labels.columns)
    print((labels['End'] - labels['Start']).describe())
    print('')
    print(labels.head())
