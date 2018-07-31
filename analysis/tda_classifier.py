import sklearn_tda as tda

import matplotlib.pyplot as plt

import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import seaborn as sn

import numpy as np
import pandas as pd
import itertools
import logging

from .persistence import Persistence
from .chunk_visualiser import ChunkVisualiser

from util import COCOKeypoints, coco_connections
from transforms import TranslateChunks, SmoothChunks, FlattenTo3D


class TDAClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, cross_validate=False):
        self.cross_validate = cross_validate
        self.arm_keypoints = [
            COCOKeypoints.RShoulder.value,
            COCOKeypoints.LShoulder.value,
            COCOKeypoints.RElbow.value,
            COCOKeypoints.LElbow.value,
            COCOKeypoints.RWrist.value,
            COCOKeypoints.LWrist.value
        ]
        self.arm_connections = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (4, 5)]
        self.all_keypoints = range(18)

    def fit(self, X, y, **fit_params):
        if self.cross_validate:
            logging.debug("Cross-validating to find best model.")
            model = self._cross_validate_pipeline()
            self.model = model.fit(X, y)
            print(self.model.best_params_)
        else:
            logging.debug("Using pre-validated pipeline.")
            model = self._pre_validated_pipeline()
            self.model = model.fit(X, y)

        # logging.info("Train accuracy = {}".format(self.model.score(X, y)))

        return self

    def predict(self, X):
        return self.model.predict(X)

    def plot_confusion_matrix(self, labels, test_labels, le):
        confusion_matrix = metrics.confusion_matrix(test_labels, labels).astype(np.float32)
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
        class_names = list(le.classes_)
        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.show(block=False)

    def _pre_validated_pipeline(self):
        pipe = Pipeline([
            ("Translate", TranslateChunks()),
            ("Smoothing", SmoothChunks()),
            ("Flattening", FlattenTo3D(self.arm_keypoints, self.arm_connections, True)),
            ("Persistence", Persistence()),
            ("Separator", tda.DiagramSelector(limit=np.inf, point_type="finite")),
            ("TDA",       tda.SlicedWasserstein(bandwidth=1.0, num_directions=10)),
            ("Estimator", SVC(kernel='precomputed'))
        ])

        return pipe

    def _cross_validate_pipeline(self):
        # Definition of pipeline
        pipe = Pipeline([
            ("Translate", TranslateChunks()),
            ("Smoothing", SmoothChunks()),
            ("Flattening", FlattenTo3D(self.arm_keypoints, self.arm_connections, True)),
            ("Persistence", Persistence()),
            ("Separator", tda.DiagramSelector(limit=np.inf, point_type="finite")),
            ("TDA",       tda.SlicedWasserstein(bandwidth=1.0, num_directions=10)),
            ("Estimator", SVC(kernel='precomputed'))
        ])

        params = [
            {
                "Smoothing": [None, SmoothChunks()],
                "Flattening__interpolate_points": [True, False],
                "Flattening__selected_keypoints": [self.arm_keypoints],
                "Flattening__connect_keypoints": [self.arm_connections]
            },
            {
                "Smoothing": [None, SmoothChunks()],
                "Flattening__interpolate_points": [False],
                "Flattening__selected_keypoints": [self.arm_keypoints, self.all_keypoints]
            },
            {
                "Smoothing": [None, SmoothChunks()],
                "Flattening__interpolate_points": [True],
                "Flattening__selected_keypoints": [self.all_keypoints],
                "Flattening__connect_keypoints": [coco_connections],
            }
        ]

        return GridSearchCV(pipe, params, n_jobs=3)

    def visualise_incorrect_classifications(self, pred_labels, test_labels, le, chunks, frames, translated_chunks, videos):
        visualiser = ChunkVisualiser(chunks, frames, translated_chunks)
        unique_labels = set(pred_labels)
        for pred_label in unique_labels:
            for true_label in unique_labels:
                if pred_label == -1 or true_label == -1 or pred_label == true_label:
                    continue

                pred_class_member_mask = (pred_labels == pred_label)
                true_class_member_mask = (test_labels == true_label)
                node = np.where(pred_class_member_mask & true_class_member_mask)[0]
                name = "P {}, T {}".format(le.classes_[pred_label], le.classes_[true_label])

                if len(node) != 0:
                    visualiser.draw_node(videos, name, node)
