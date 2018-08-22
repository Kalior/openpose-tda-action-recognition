import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from ..util import COCOKeypoints, coco_connections
from .. import classifiers, transforms
from . import AverageSpeed, AngleChangeSpeed, AmountOfMovement, KeypointDistance
from ..analysis import ChunkVisualiser


class FeatureVisualiser:
    """Visaliser for features.

    """

    def visualise_point_cloud(self, chunks):
        """Plots the point clouds that are given to the persistence calculator.

        Does not plot one of the features in the features package, but instead
        the feature that is created by combining most of the transforms from
        the transforms package.

        Parameters
        ----------
        chunks : array-like, shape = [n_chunks, frames_per_chunk, n_keypoints, 3]
            The training data for classification.
        """
        arm_keypoints = [k.value for k in [
            COCOKeypoints.Neck,
            COCOKeypoints.RWrist,
            COCOKeypoints.LWrist,
            COCOKeypoints.RAnkle,
            COCOKeypoints.LAnkle,
        ]]
        arm_connections = [(0, 1), (0, 2), (1, 3), (2, 4)]  # , (0, 4), (4, 5), (5, 3)]
        pipe = Pipeline([
            ("1", transforms.ExtractKeypoints(arm_keypoints)),
            ("3", transforms.SmoothChunks()),
            # ("4", transforms.InterpolateKeypoints(arm_connections, 1)),
            ("2", transforms.TranslateChunks()),
            # ("6", transforms.Speed()),
            ("5", transforms.FlattenTo3D()),
        ])
        chunks = pipe.fit_transform(chunks)
        transforms.Persistence().visualise_point_clouds(chunks, 10)

    def save_persistence_graphs(self, chunks, labels, out_dir):
        """Saves the persistence graphs corresponding to the chunks to out_dir.

        Parameters
        ----------
        chunks : array-like, shape = [n_chunks, frames_per_chunk, n_keypoints, 3]
            The training data for classification.
        """
        arm_keypoints = [k.value for k in [
            COCOKeypoints.Neck,
            COCOKeypoints.RWrist,
            COCOKeypoints.LWrist,
            COCOKeypoints.RAnkle,
            COCOKeypoints.LAnkle,
        ]]
        pipe = Pipeline([
            ("1", transforms.ExtractKeypoints(arm_keypoints)),
            ("2", transforms.SmoothChunks()),
            ("3", transforms.TranslateChunks()),
            ("4", transforms.FlattenTo3D()),
        ])
        chunks = pipe.fit_transform(chunks)
        persistence = transforms.Persistence()
        _ = persistence.fit_transform(chunks)
        persistence.save_persistences(labels, out_dir)
        persistence.save_betti_curves(labels, out_dir)

    def visualise_features(self, chunks, labels):
        """Plots all of the features from this package, divided by label.

        Uses the parameters from feature classifier for the features.

        Parameters
        ----------
        chunks : array-like, shape = [n_chunks, frames_per_chunk, n_keypoints, 3]
            The training data for classification.
        labels : array-like, shape = [n_chunks, 1]
            The training labels.
        """
        features = classifiers.FeatureEngineeringClassifier(True)

        pipe = features._feature_engineering_union()
        self._plot_feature_per_class(chunks, pipe, labels, 'Combined features')

        angle_change_connections = features.angle_change_connections
        speed_keypoints = features.speed_keypoints
        keypoint_distance_connections = features.keypoint_distance_connections

        chunk_speed = AverageSpeed(speed_keypoints)
        self._plot_feature_per_class(chunks, chunk_speed, labels, 'Average Speed')

        angle_change_speed = AngleChangeSpeed(angle_change_connections)
        self._plot_feature_per_class(chunks, angle_change_speed, labels, 'Average Angle Change')

        movement = AmountOfMovement(range(18))
        self._plot_feature_per_class(chunks, movement, labels, 'Total distance')

        keypoint_distance = KeypointDistance(keypoint_distance_connections)
        self._plot_feature_per_class(chunks, keypoint_distance, labels, 'Keypoint distances')
        plt.show(block=False)

    def _plot_feature_per_class(self, chunks, transform, labels, title):
        feature = Pipeline([
            ("Feature", transform),
            ("Scaler", RobustScaler())
        ]).fit_transform(chunks, labels)

        logging.debug('Constructing dataframe')
        rows = [{'value': feature[i, j], 'keypoint': j, 'action': labels[i]}
                for i in range(feature.shape[0]) for j in range(feature.shape[1])]
        df = pd.DataFrame(rows, columns=['value', 'keypoint', 'action'])

        logging.debug('Preparing plot.')
        plt.figure()
        sns.lineplot(x='keypoint', y='value', hue='action', style=None, data=df)
        plt.title(title)

    def visualise_classes(self, train, test):
        """Draws the average shape of the input data.

        Not a feature in the feature package, but gives an indication of
        what can be used for features.

        Parameters
        ----------
        chunks : array-like, shape = [n_chunks, frames_per_chunk, n_keypoints, 3]
            The training data for classification.
        frames : array-like, shape = [n_chunks, frames_per_chunk, 1]
            The frame numbers for the chunks.
        labels : array-like, shape = [n_chunks, 1]
            The training labels.

        """

        chunks, frames, labels = self._append_train_and_test(train, test)

        translated_chunks = transforms.TranslateChunks().transform(chunks)
        visualiser = ChunkVisualiser(chunks, frames, translated_chunks)
        unique_labels = set(labels)
        nodes = {}
        for k in unique_labels:
            class_member_mask = (labels == k)
            node = np.where(class_member_mask)[0]
            name = str(k)
            nodes[name] = node

            print("{}: {}".format(k, np.mean([c.shape[0] for c in chunks[node]])))

        visualiser.visualise_averages(nodes, True)

    def _append_train_and_test(self, train, test):
        chunks = np.append(train[0], test[0], axis=0)
        frames = np.append(train[1], test[1], axis=0)
        labels = np.append(train[2], test[2], axis=0)
        return chunks, frames, labels
