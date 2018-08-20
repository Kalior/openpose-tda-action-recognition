import argparse
import logging
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from action_recognition.classifiers import TDAClassifier, EnsembleClassifier, \
    ClassificationVisualiser, FeatureEngineeringClassifier
from action_recognition import transforms
from action_recognition.features import FeatureVisualiser
from action_recognition.util import COCOKeypoints, coco_connections
from action_recognition import augmentors


def main(args):
    extension_free = os.path.splitext(args.dataset)[0]
    train_name = extension_free + '-train.npz'
    test_name = extension_free + '-test.npz'
    train = load_data(train_name)
    test = load_data(test_name)

    logging.info("Number of train dataset labels: {}".format(Counter(train[2])))
    logging.info("Number of test dataset labels: {}".format(Counter(test[2])))

    train = augmentors.Rotate(2).augment(*train)
    logging.info("Number of train dataset labels after augmentor: {}".format(Counter(train[2])))

    if args.tda:
        classifier = TDAClassifier(cross_validate=args.cross_validate)
        run_classifier(train, test, args.title, classifier)
    if args.ensemble:
        classifier = EnsembleClassifier()
        run_classifier(train, test, args.title, classifier)
    if args.feature_engineering:
        classifier = FeatureEngineeringClassifier()
        run_classifier(train, test, args.title, classifier)
    if args.visualise:
        vis = FeatureVisualiser()
        vis.visualise_features(train[0], train[2])
        vis.visualise_point_cloud(train[0])
        vis.visualise_classes(train, test)
        plt.show()


def load_data(file_name):
    dataset_npz = np.load(file_name)
    chunks = dataset_npz['chunks']
    frames = dataset_npz['frames']
    labels = dataset_npz['labels']
    videos = dataset_npz['videos']

    return chunks, frames, labels, videos


def run_classifier(train, test, title, classifier):
    train_chunks, _, train_labels, _ = train
    test_chunks, test_frames, test_labels, test_videos = test
    test_translated_chunks = transforms.TranslateChunks().transform(test_chunks)

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.transform(test_labels)

    logging.info("Fitting classifier.")
    classifier.fit(train_chunks, train_labels)
    logging.info("Predicting classes of test data.")
    pred_labels = classifier.predict(test_chunks)

    accuracy = metrics.accuracy_score(test_labels, pred_labels)
    precision = metrics.precision_score(test_labels, pred_labels, average='weighted')
    recall = metrics.recall_score(test_labels, pred_labels, average='weighted')

    logging.info("Accuracy: {:.3f}\nPrecision: {:.3f}\nRecall: {:.3f}".format(
        accuracy, precision, recall))

    file_name = "{}.pkl".format(title)
    logging.info("Saving model to {}.".format(file_name))
    joblib.dump(classifier, file_name)

    visualiser = ClassificationVisualiser()
    visualiser.plot_confusion_matrix(pred_labels, test_labels, le, title)
    visualiser.visualise_incorrect_classifications(
        pred_labels, test_labels, le, test_chunks, test_frames, test_translated_chunks, test_videos)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TDA analysis of the tracks.')
    parser.add_argument('--dataset', type=str, help='The path to the dataset')

    parser.add_argument('--tda', action='store_true',
                        help='Run a TDA algorithm on the data.')
    parser.add_argument('--ensemble', action='store_true',
                        help='Runs a voting classifier on TDA and feature engineering on the dataset.')
    parser.add_argument('--feature-engineering', action='store_true',
                        help='Runs a classifier by using feature engineering.')

    parser.add_argument('--visualise', action='store_true',
                        help='Specify if you wish to only visualise the classes in the dataset.')
    parser.add_argument('--title', type=str, default='classifier',
                        help='Title and file name for confusion matrix plot.')
    parser.add_argument('--cross-validate', '-cv', action='store_true',
                        help='Specify for cross-validation of tda pipeline.')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
