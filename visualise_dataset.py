import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from action_recognition.features import FeatureVisualiser
from action_recognition.util import load_data


def main(args):
    extension_free = os.path.splitext(args.dataset)[0]
    train_name = extension_free + '-train.npz'
    test_name = extension_free + '-test.npz'
    train = load_data(train_name)
    test = load_data(test_name)

    if args.persistence_graphs:
        FeatureVisualiser().save_persistence_graphs(train[0], train[2], args.out_directory)
    if args.point_clouds:
        FeatureVisualiser().visualise_point_cloud(train[0])
    if args.features:
        FeatureVisualiser().visualise_features(train[0], train[2])
    if args.classes:
        FeatureVisualiser().visualise_classes(train, test)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Helps visualise the dataset, e.g. by showing the features or the point clouds.'))
    parser.add_argument('--dataset', type=str,
                        help=('The path to the dataset, will append -train.npz and -test.npz '
                              'to this path, so make sure to not include those.'))

    parser.add_argument('--out-directory', type=str, default='output/graphs',
                        help='Path to where the persistence graphs are saved if wanted.')

    parser.add_argument('--persistence-graphs', action='store_true',
                        help='Saves the persistence graphs of the dataset to file.')
    parser.add_argument('--point-clouds', action='store_true',
                        help='Uses matplotlib\'s 3D plotting to display the point clouds of the data.')
    parser.add_argument('--features', action='store_true',
                        help='Draws the features that are used in the feature engineering.')
    parser.add_argument('--classes', action='store_true',
                        help='Draws the average shape of each class in the data.')

    args = parser.parse_args()

    main(args)
