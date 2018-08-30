import argparse
import logging
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.externals import joblib

from action_recognition.classifiers import TDAClassifier, EnsembleClassifier, \
    ClassificationVisualiser, FeatureEngineeringClassifier
from action_recognition import transforms
from action_recognition import augmentors
from action_recognition.util import load_data


def main(args):
    name, extension = os.path.splitext(args.dataset)
    train_name = name + '-train.npz'
    test_name = name + '-test.npz'
    train = load_data(train_name)
    test = load_data(test_name)

    logging.info("Number of train dataset labels: {}".format(Counter(train[2])))
    logging.info("Number of test dataset labels: {}".format(Counter(test[2])))

    if args.augmentation:
        train = augmentors.Rotate(args.augmentation_amount).augment(*train)
        logging.info(
            "Number of train dataset labels after augmentor: {}".format(Counter(train[2])))

    if args.tda:
        classifier = TDAClassifier(cross_validate=args.cross_validate)
    elif args.ensemble:
        classifier = EnsembleClassifier(use_tda_vectorisations=args.use_tda_vectorisations)
    elif args.feature_engineering:
        classifier = FeatureEngineeringClassifier(
            use_tda_vectorisations=args.use_tda_vectorisations)

    train_classifier(train, test, args.title, classifier, args.visualise_incorrect_classifications)


def train_classifier(train, test, title, classifier, visualise_incorrect_classifications):
    train_chunks, _, train_labels, _ = train
    test_chunks, test_frames, test_labels, test_videos = test
    test_translated_chunks = transforms.TranslateChunks().transform(test_chunks)

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
    visualiser.plot_confusion_matrix(pred_labels, test_labels, classifier.classes_, title)

    if visualise_incorrect_classifications:
        visualiser.visualise_incorrect_classifications(
            pred_labels, test_labels, test_chunks, test_frames, test_translated_chunks, test_videos)

    file_name = "{}.pkl".format(title)
    logging.info("Saving model to {}.".format(file_name))
    joblib.dump(classifier, file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Trains a classifier on recognising actions from the given dataset. '
        'Requires one of --tda, --ensemble or --feature-engineering.'))
    parser.add_argument('--dataset', type=str,
                        help=('The path to the dataset, will append -train.npz and -test.npz '
                              'to this path, so make sure to not include those.'))

    parser.add_argument('--tda', action='store_true',
                        help='Run a TDA algorithm on the data.')
    parser.add_argument('--ensemble', action='store_true',
                        help='Runs a voting classifier on TDA and feature engineering on the dataset.')
    parser.add_argument('--feature-engineering', action='store_true',
                        help='Runs a classifier by using feature engineering.')

    parser.add_argument('--title', type=str, default='classifier',
                        help=('Title and file name for confusion matrix plot '
                              'as well as the name of the .pkl classifier file.'))
    parser.add_argument('--cross-validate', '-cv', action='store_true',
                        help=('Specify for cross-validation of tda pipeline. '
                              'In order to change what is cross-validated, see '
                              'action_recognition.classifiers.tda_classifier '))

    parser.add_argument('--visualise-incorrect-classifications', action='store_true',
                        help=('If specified, the actions that are incorrectly classified will be '
                              'drawn to help get an understanding of what the classifier misses.'))

    parser.add_argument('--use-tda-vectorisations', action='store_true',
                        help=('Specify for if the feature engineering and ensemble classifiers '
                              'should make use of the tda vectorisations from sklearn_tda. '
                              'Note that this will cause the model saving to file to crash '
                              'since parts of the sklearn_tda vectorisations are not pickable.'))

    parser.add_argument('--augmentation', action='store_true',
                        help=('Specify for if the training data should be augmented. '
                              'Only current augmentor is a rotation augmentor.'))
    parser.add_argument('--augmentation-amount', type=int, default=2,
                        help='Number of points to add during augmentation.')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
