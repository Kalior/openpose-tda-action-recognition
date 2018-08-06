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
from sklearn.pipeline import Pipeline

from action_recognition.analysis import Mapper, ChunkVisualiser
from action_recognition.classifiers import TDAClassifier, EnsembleClassifier, ClassificationVisualiser
from action_recognition import transforms
from action_recognition import features
from action_recognition.util import COCOKeypoints, coco_connections


def main(args):
    extension_free = os.path.splitext(args.dataset)[0]
    train_name = extension_free + '-train.npz'
    test_name = extension_free + '-test.npz'
    train = load_data(train_name)
    test = load_data(test_name)

    logging.info("Number of train dataset labels: {}".format(Counter(train[2])))
    logging.info("Number of test dataset labels: {}".format(Counter(test[2])))

    if args.tda:
        run_tda(train, test, args.title)
    if args.mapper:
        run_mapper(train, test)
    if args.ensemble:
        run_ensemble(train, test, args.title)
    if args.visualise:
        visualise_features(train[0], train[2])
        visualise_classes(train, test)
        visualise_point_cloud(train, test)
        plt.show()


def load_data(file_name):
    dataset_npz = np.load(file_name)
    chunks = dataset_npz['chunks']
    frames = dataset_npz['frames'].astype(np.int)
    labels = dataset_npz['labels']
    videos = dataset_npz['videos']

    return chunks, frames, labels, videos


def visualise_point_cloud(train):
    arm_keypoints = [
        COCOKeypoints.RShoulder.value,
        COCOKeypoints.LShoulder.value,
        COCOKeypoints.RElbow.value,
        COCOKeypoints.LElbow.value,
        COCOKeypoints.RWrist.value,
        COCOKeypoints.LWrist.value
    ]
    arm_connections = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (4, 5)]
    chunks = train[0]
    pipe = Pipeline([
        ("1", transforms.TranslateChunks()),
        ("2", transforms.SmoothChunks()),
        ("3", transforms.ExtractKeypoints(arm_keypoints)),
        ("4", transforms.InterpolateKeypoints(arm_connections)),
        ("5", transforms.FlattenTo3D()),
        ("6", transforms.RotatePointCloud(2))
    ])
    chunks = pipe.fit_transform(chunks)
    transforms.Persistence().visualise_point_clouds(chunks, 10)


def visualise_features(chunks, labels):
    chunk_speed = features.AverageSpeed(range(18)).transform(chunks)
    plot_feature_per_class(chunk_speed, labels, 'Average Speed')
    angle_change_speed = features.AngleChangeSpeed(coco_connections).transform(chunks)
    plot_feature_per_class(angle_change_speed, labels, 'Average Angle Change')
    movement = features.AmountOfMovement(range(18)).transform(chunks)
    plot_feature_per_class(movement, labels, 'Total distance')
    connections = [
        (COCOKeypoints.RWrist.value, COCOKeypoints.LWrist.value),
        (COCOKeypoints.RElbow.value, COCOKeypoints.LElbow.value),
        (COCOKeypoints.Neck.value, COCOKeypoints.LAnkle.value),
        (COCOKeypoints.Neck.value, COCOKeypoints.RAnkle.value),
        (COCOKeypoints.LWrist.value, COCOKeypoints.LAnkle.value),
        (COCOKeypoints.RWrist.value, COCOKeypoints.RAnkle.value)
    ]
    distances = features.KeypointDistance(connections).transform(chunks)
    plot_feature_per_class(distances, labels, 'Keypoint distances')
    plt.show(block=False)


def plot_feature_per_class(feature, labels, title):
    logging.debug('Constructing dataframe')
    rows = [{'value': feature[i, j], 'keypoint': j, 'action': labels[i]}
            for i in range(feature.shape[0]) for j in range(feature.shape[1])]
    df = pd.DataFrame(rows, columns=['value', 'keypoint', 'action'])

    logging.debug('Preparing plot.')
    plt.figure()
    sns.lineplot(x='keypoint', y='value', hue='action', style='action', data=df)
    plt.title(title)


def append_train_and_test(train, test):
    chunks = np.append(train[0], test[0], axis=0)
    frames = np.append(train[1], test[1], axis=0)
    labels = np.append(train[2], test[2], axis=0)
    videos = np.append(train[3], test[3], axis=0)
    return chunks, frames, labels, videos


def visualise_classes(train, test):
    chunks, frames, labels, videos = append_train_and_test(train, test)
    translated_chunks = transforms.TranslateChunks().transform(chunks)
    visualiser = ChunkVisualiser(chunks, frames, translated_chunks)
    unique_labels = set(labels)
    nodes = {}
    for k in unique_labels:
        class_member_mask = (labels == k)
        node = np.where(class_member_mask)[0]
        name = str(k)
        nodes[name] = node

    visualiser.visualise_averages(nodes, True)


def run_ensemble(train, test, title):
    train_chunks, _, train_labels, _ = train
    test_chunks, test_frames, test_labels, test_videos = test
    test_translated_chunks = transforms.TranslateChunks().transform(test_chunks)

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.transform(test_labels)

    classifier = EnsembleClassifier()
    logging.info("Fitting classifier.")
    classifier.fit(train_chunks, train_labels)
    logging.info("Predicting classes of test data.")
    pred_labels = classifier.predict(test_chunks)

    accuracy = metrics.accuracy_score(test_labels, pred_labels)
    precision = metrics.precision_score(test_labels, pred_labels, average='weighted')
    recall = metrics.recall_score(test_labels, pred_labels, average='weighted')

    logging.info("Accuracy: {:.3f}\nPrecision: {:.3f}\nRecall: {:.3f}".format(
        accuracy, precision, recall))

    visualiser = ClassificationVisualiser()
    visualiser.plot_confusion_matrix(pred_labels, test_labels, le, title)
    visualiser.visualise_incorrect_classifications(
        pred_labels, test_labels, le, test_chunks, test_frames, test_translated_chunks, test_videos)


def run_tda(train, test, title):
    train_chunks, _, train_labels, _ = train
    test_chunks, test_frames, test_labels, test_videos = test
    test_translated_chunks = transforms.TranslateChunks().transform(test_chunks)

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.transform(test_labels)

    classifier = TDAClassifier(cross_validate=False)
    logging.info("Fitting classifier.")
    classifier.fit(train_chunks, train_labels)
    logging.info("Predicting classes of test data.")
    pred_labels = classifier.predict(test_chunks)

    accuracy = metrics.accuracy_score(test_labels, pred_labels)
    precision = metrics.precision_score(test_labels, pred_labels, average='weighted')
    recall = metrics.recall_score(test_labels, pred_labels, average='weighted')

    logging.info("Accuracy: {:.3f}\nPrecision: {:.3f}\nRecall: {:.3f}".format(
        accuracy, precision, recall))

    visualiser = ClassificationVisualiser()
    visualiser.plot_confusion_matrix(pred_labels, test_labels, le, title)
    visualiser.visualise_incorrect_classifications(
        pred_labels, test_labels, le, test_chunks, test_frames, test_translated_chunks, test_videos)


def run_mapper(test, train):
    chunks, frames, labels, videos = append_train_and_test(train, test)
    translated_chunks = transforms.TranslateChunks().transform(chunks)

    logging.info("Smoothing chunks.")
    translated_chunks = transforms.SmoothChunks().transform(translated_chunks)

    selected_keypoints = [
        COCOKeypoints.RShoulder.value,
        COCOKeypoints.LShoulder.value,
        COCOKeypoints.RElbow.value,
        COCOKeypoints.LElbow.value,
        COCOKeypoints.RWrist.value,
        COCOKeypoints.LWrist.value
    ]
    logging.info("Flattening data into a datapoint per chunk.")
    data = transforms.Flatten(selected_keypoints).transform(translated_chunks)
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
    parser.add_argument('--ensemble', action='store_true',
                        help='Runs a voting classifier on TDA and feature engineering on the dataset.')
    parser.add_argument('--title', type=str, default='Confusion matrix',
                        help='Title and file name for confusion matrix plot.')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
