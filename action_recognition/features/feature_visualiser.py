import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from ..util import COCOKeypoints
from .. import transforms
from . import AverageSpeed, AngleChangeSpeed, AmountOfMovement, KeypointDistance
from ..analysis import ChunkVisualiser
from .. import classifiers


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
        arm_keypoints = [
            COCOKeypoints.RElbow.value,
            COCOKeypoints.RWrist.value,
            COCOKeypoints.LElbow.value,
            COCOKeypoints.LWrist.value,
            COCOKeypoints.LKnee.value,
            COCOKeypoints.LAnkle.value,
            COCOKeypoints.RKnee.value,
            COCOKeypoints.RAnkle.value
        ]
        arm_connections = [(0, 1), (2, 3), (4, 5), (6, 7)]
        pipe = Pipeline([
            ("1", transforms.TranslateChunks()),
            ("2", transforms.ExtractKeypoints(arm_keypoints)),
            ("3", transforms.SmoothChunks()),
            ("4", transforms.InterpolateKeypoints(arm_connections)),
            ("5", transforms.FlattenTo3D()),
        ])
        chunks = pipe.fit_transform(chunks)
        transforms.Persistence().visualise_point_clouds(chunks, 10)

    def visualise_features(self, chunks, labels):
        """Plots all of the features from this package, divided by label.

        Uses the parameters from ensemble classifier for the features.

        Parameters
        ----------
        chunks : array-like, shape = [n_chunks, frames_per_chunk, n_keypoints, 3]
            The training data for classification.
        labels : array-like, shape = [n_chunks, 1]
            The training labels.
        """
        ensemble = classifiers.EnsembleClassifier()
        angle_change_connections = ensemble.angle_change_connections
        speed_keypoints = ensemble.speed_keypoints
        keypoint_distance_connections = ensemble.keypoint_distance_connections

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
        ]).fit_transform(chunks)

        logging.debug('Constructing dataframe')
        rows = [{'value': feature[i, j], 'keypoint': j, 'action': labels[i]}
                for i in range(feature.shape[0]) for j in range(feature.shape[1])]
        df = pd.DataFrame(rows, columns=['value', 'keypoint', 'action'])

        logging.debug('Preparing plot.')
        plt.figure()
        sns.lineplot(x='keypoint', y='value', hue='action', style=None, data=df)
        plt.title(title)

    def visualise_classes(self, chunks, frames, labels):
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
