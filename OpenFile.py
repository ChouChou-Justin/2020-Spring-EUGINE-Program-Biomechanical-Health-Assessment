import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def openFile(path):
    f = pd.read_csv(path)
    f_shuffle = shuffle(f)
    classes = f_shuffle['class']
    features = f_shuffle.drop('class', axis='columns')

    featureColumns = []
    for feature in features.columns:
        featureColumns.append(feature)

    return features, classes, featureColumns

def DenoteFile(path):
    f = pd.read_csv(path)
    label = np.zeros(len(f))
    for i in range(len(f['class'])):
        if f['class'][i] == 'Abnormal':
            label[i] = 1
        else:
            label[i] = 0
    f.insert(7, 'label', label)
    dataset = f.drop(columns='class')
    dataset_shuffle = shuffle(dataset)
    labels = dataset_shuffle['label']
    features = dataset_shuffle.drop('label', axis='columns')

    featureColumns = []
    for feature in features.columns:
        featureColumns.append(feature)

    return dataset_shuffle, labels, featureColumns
