import argparse
import logging
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

from action_recognition.classifiers import TDAClusterer
from action_recognition.util import load_data


def main(args):
    extension_free = os.path.splitext(args.dataset)[0]
    train_name = extension_free + '-train.npz'
    test_name = extension_free + '-test.npz'
    train = load_data(train_name)
    test = load_data(test_name)

    logging.info("Number of train dataset labels: {}".format(Counter(train[2])))
    logging.info("Number of test dataset labels: {}".format(Counter(test[2])))

    train_clusterer(train, test)


def append_train_and_test(train, test):
    chunks = np.append(train[0], test[0], axis=0)
    frames = np.append(train[1], test[1], axis=0)
    labels = np.append(train[2], test[2], axis=0)
    videos = np.append(train[3], test[3], axis=0)
    return chunks, frames, labels, videos


def train_clusterer(test, train):
    chunks, frames, labels, videos = append_train_and_test(train, test)

    le = LabelEncoder()
    enc_labels = le.fit_transform(labels)

    clusterer = TDAClusterer()
    pred_labels, x_transformed = clusterer.fit_predict(chunks)

    clusterer.plot_clusters(x_transformed, pred_labels, 'Estimated')
    clusterer.plot_clusters(x_transformed, labels, 'True')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TDA clustering of the dataset.')
    parser.add_argument('--dataset', type=str, help='The path to the dataset')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
