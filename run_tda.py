import argparse
import logging
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

from tracker import Person, Track, TrackVisualiser
from analysis import Mapper, TDAClassifier, ChunkVisualiser
from transforms import Flatten, FlattenTo3D, SmoothChunks, \
    TranslateChunks, TranslateChunksByKeypoints, AverageSpeed, AngleChangeSpeed, AmountOfMovement
from util import COCOKeypoints, coco_connections


def main(args):
    dataset_npz = np.load(args.dataset)
    chunks = dataset_npz['chunks']
    frames = dataset_npz['frames'].astype(np.int)
    labels = dataset_npz['labels']
    videos = dataset_npz['videos']

    logging.info("Number of dataset labels: {}".format(Counter(labels)))

    logging.info("Translating every chunk by the average position of that chunk.")
    translated_chunks = TranslateChunks().transform(chunks)

    if args.tda:
        run_tda(chunks, frames, translated_chunks, videos, labels)
    if args.mapper:
        run_mapper(chunks, frames, translated_chunks, videos, labels)
    if args.visualise:
        chunk_speed = AverageSpeed(range(18)).transform(chunks)
        angle_change_speed = AngleChangeSpeed(coco_connections).transform(chunks)
        movement = AmountOfMovement(range(18)).transform(chunks)
        plot_feature_per_class(movement, labels)
        visualise_classes(chunks, frames, translated_chunks, labels)


def plot_feature_per_class(feature, labels):
    logging.debug('Constructing dataframe')
    df = pd.DataFrame(columns=['speed', 'keypoint', 'action'])
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            row = {
                'speed': feature[i, j],
                'keypoint': j,
                'action': labels[i]
            }
            df = df.append(row, ignore_index=True)

    logging.debug('Preparing plot.')
    sns.lineplot(x='keypoint', y='speed', hue='action', style='action', data=df)
    plt.show()


def visualise_classes(chunks, frames, translated_chunks, labels):
    visualiser = ChunkVisualiser(chunks, frames, translated_chunks)
    unique_labels = set(labels)
    nodes = {}
    for k in unique_labels:
        if k == -1:
            continue

        class_member_mask = (labels == k)
        node = np.where(class_member_mask)[0]
        name = str(k)
        nodes[name] = node

    visualiser.visualise_averages(nodes, True)


def run_tda(chunks, frames, translated_chunks, videos, labels):
    le = LabelEncoder()
    enc_labels = le.fit_transform(labels)
    logging.debug("Splitting data into test/train")
    train_chunks, test_chunks, train_labels, test_labels, \
        _, test_frames, \
        _, test_videos, \
        _, test_translated_chunks = train_test_split(
            chunks, enc_labels, frames, videos, translated_chunks)

    classifier = TDAClassifier(cross_validate=False)
    classifier.fit(train_chunks, train_labels)
    pred_labels = classifier.predict(test_chunks)

    accuracy = metrics.accuracy_score(test_labels, pred_labels)
    precision = metrics.precision_score(test_labels, pred_labels, average='weighted')
    recall = metrics.recall_score(test_labels, pred_labels, average='weighted')

    logging.info("Accuracy: {:.3f}\nPrecision: {:.3f}\nRecall: {:.3f}".format(
        accuracy, precision, recall))

    classifier.plot_confusion_matrix(pred_labels, test_labels, le)
    classifier.visualise_incorrect_classifications(
        pred_labels, test_labels, le, test_chunks, test_frames, test_translated_chunks, test_videos)


def run_mapper(chunks, frames, translated_chunks, videos, labels):
    logging.info("Smoothing chunks.")
    translated_chunks = SmoothChunks().transform(translated_chunks)

    selected_keypoints = [
        COCOKeypoints.RShoulder.value,
        COCOKeypoints.LShoulder.value,
        COCOKeypoints.RElbow.value,
        COCOKeypoints.LElbow.value,
        COCOKeypoints.RWrist.value,
        COCOKeypoints.LWrist.value
    ]
    logging.info("Flattening data into a datapoint per chunk.")
    data = Flatten(selected_keypoints).transform(translated_chunks)
    logging.info("Applying mapping to tracks.")
    mapper = Mapper(chunks, frames, translated_chunks, labels)
    mapper.create_tooltips(videos)
    graph = mapper.mapper(data)
    logging.info("Visualisation of the resulting nodes.")
    mapper.visualise(videos, graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TDA analysis of the tracks.')
    parser.add_argument('--dataset', type=str, help='The path to the dataset')
    parser.add_argument('--mapper', action='store_true',
                        help='Run the mapper algorithm on the data')
    parser.add_argument('--tda', action='store_true',
                        help='Run a TDA algorithm on the data.')
    parser.add_argument('--visualise', action='store_true',
                        help='Specify if you wish to only visualise the classes in the dataset.')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
