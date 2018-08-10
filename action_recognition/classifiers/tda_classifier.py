import sklearn_tda as tda

import matplotlib.pyplot as plt

import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import KernelCenterer

import numpy as np
import itertools
import logging

from ..util import COCOKeypoints, coco_connections
from ..transforms import Persistence, TranslateChunks, SmoothChunks, FlattenTo3D, Speed, \
    ExtractKeypoints, InterpolateKeypoints, RotatePointCloud


class TDAClassifier(BaseEstimator, ClassifierMixin):
    """Classifier for actions.

    Uses Gudhi and sklearn_tda under the hood.
    Uses a pipeline where the input chunks are transformed into
    3D-point clouds (with time as the 3rd dimension) upon which
    persistence diagrams are calculated.  The persistence diagrams
    are then passed into the kernel sklearn_tda.SlicedWasserstein,
    and finally a sklearn.SVC is fitted to the data.

    Parameters
    ----------
    cross_validate : boolean, optional
        Specifies if the model should be cross validated in order to find the
        best parameters for the input data, or if a previously determined
        best model should be used.
    """

    def __init__(self, cross_validate=False):
        self.cross_validate = cross_validate
        self.arm_keypoints = [
            COCOKeypoints.RElbow.value,
            COCOKeypoints.RWrist.value,
            COCOKeypoints.LElbow.value,
            COCOKeypoints.LWrist.value,
            COCOKeypoints.LAnkle.value,
            COCOKeypoints.RAnkle.value
        ]
        self.arm_connections = [(0, 1), (2, 3), (4, 5)]
        self.all_keypoints = range(18)

    def fit(self, X, y, **fit_params):
        """Fit the model.

        Fits the model by using the pipeline of transforms going from
        chunks into a 3D-point cloud, and then calculating the persistence
        and the Sliced Wasserstein kernel, finally training a SVC.

        Parameters
        ----------
        X : iterable
            Training data, must be passable to transforms.TranslateChunks()
        y : iterable
            Training labels.
        fit_params : dict
            ignored for now.

        """

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
        """Predicts using the pipeline.

        Parameters
        ----------
        X : iterable
            Data to predict labels for.
            Must be passable to transforms.TranslateChunks()

        Returns
        -------
        y_pred : array-like

        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predicts using the pipeline.

        Parameters
        ----------
        X : iterable
            Data to predict labels for.
            Must be passable to transforms.TranslateChunks()

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]

        """
        return self.model.predict_proba(X)

    def _pre_validated_pipeline(self):
        pipe = Pipeline([
            ("Translate",   TranslateChunks()),
            ("Extract",     ExtractKeypoints(self.arm_keypoints)),
            ("Smoothing",   SmoothChunks()),
            ("Interpolate", InterpolateKeypoints(self.arm_connections, 2)),
            ("Flattening",  FlattenTo3D()),
            ("Persistence", Persistence(max_edge_length=0.9, complex_='rips')),
            ("Separator",   tda.DiagramSelector(limit=np.inf, point_type="finite")),
            ("TDA",         tda.SlicedWasserstein(bandwidth=0.6, num_directions=20)),
            ("Centerer",    KernelCenterer()),
            ("Estimator",   SVC(kernel='precomputed', probability=True))
        ])

        return pipe

    def _cross_validate_pipeline(self):
        # Definition of pipeline
        pipe = self._pre_validated_pipeline()

        params = [
            {
                "Persistence__max_edge_length": [0.5, 0.9],
                "Persistence__complex_": ['rips']
                "Extract__selected_keypoints": [self.arm_keypoints],
                "Interpolate__number_of_points": [2, 3, 4],
                "Interpolate__connect_keypoints": [arm_connections],
            },
            {
                "Persistence__max_edge_length": [0.5, 0.9],
                "Persistence__complex_": ['rips']
                "Extract__selected_keypoints": [self.all_keypoints],
                "Interpolate__connect_keypoints": [arm_connections, coco_connections],
                "Interpolate__number_of_points": [2, 3, 4],
                "Interpolate": [None, InterpolateKeypoints()]
            },
            {
                "Persistence__max_alpha_square": [0.9],
                "Persistence__complex_": ['alpha'],
                "Extract__selected_keypoints": [self.all_keypoints],
                "Interpolate__connect_keypoints": [arm_connections, coco_connections],
                "Interpolate__number_of_points": [2, 3, 4],
                "Interpolate": [None, InterpolateKeypoints()]
            },
            {
                "Persistence__max_edge_length": [0.5, 0.9],
                "Persistence__complex_": ['rips']
                "Extract__selected_keypoints": [self.all_keypoints],
                "Interpolate__connect_keypoints": [arm_connections, coco_connections],
                "Interpolate__number_of_points": [2, 3, 4],
                "Interpolate": [None, InterpolateKeypoints()]
            },
            # {
            #     "Smoothing": [SmoothChunks()],
            #     "Interpolate": [None],
            #     "Extract__selected_keypoints": [self.all_keypoints],
            #     "Interpolate__connect_keypoints": [coco_connections],
            # }
        ]

        return GridSearchCV(pipe, params, n_jobs=-1)
