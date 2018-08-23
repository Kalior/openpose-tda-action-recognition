import sklearn_tda as tda

import matplotlib.pyplot as plt

import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler

import numpy as np
import itertools
import logging

from ..util import COCOKeypoints, coco_connections
from .. import transforms


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
        self.selected_keypoints = [k.value for k in [
            COCOKeypoints.Neck,
            COCOKeypoints.RWrist,
            COCOKeypoints.LWrist,
            COCOKeypoints.RAnkle,
            COCOKeypoints.LAnkle,
        ]]
        self.keypoint_connections = [(0, 1), (0, 2), (1, 3), (2, 4)]
        self.all_keypoints = range(14)

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

        self.classes_ = self.model.classes_
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
            ("Extract", transforms.ExtractKeypoints(self.selected_keypoints)),
            ("Smoothing", transforms.SmoothChunks()),
            ("Translate", transforms.TranslateChunks()),
            ("PositionCloud", transforms.FlattenTo3D()),
            ("Persistence", transforms.Persistence(max_alpha_square=1, complex_='alpha')),
            ("Separator", tda.DiagramSelector(limit=np.inf, point_type="finite")),
            ("Prominent", tda.ProminentPoints()),
            ("TDA", tda.SlicedWasserstein(bandwidth=0.6, num_directions=20)),
            ("Estimator", SVC(kernel='precomputed', probability=True))
        ])

        return pipe

    def _cross_validate_pipeline(self):
        # Definition of pipeline
        pipe = self._pre_validated_pipeline()

        limb_connections = [
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (8, 9),
            (9, 10),
            (11, 12),
            (12, 13)
        ]

        leg_keypoints = [k.value for k in [
            COCOKeypoints.LKnee,
            COCOKeypoints.LAnkle,
            COCOKeypoints.RKnee,
            COCOKeypoints.RAnkle
        ]]
        leg_connections = [(0, 1), (2, 3)]

        leg_and_arm_keypoints = [k.value for k in [
            COCOKeypoints.RElbow,
            COCOKeypoints.RWrist,
            COCOKeypoints.LElbow,
            COCOKeypoints.LWrist,
            COCOKeypoints.LKnee,
            COCOKeypoints.LAnkle,
            COCOKeypoints.RKnee,
            COCOKeypoints.RAnkle
        ]]
        leg_and_arm_connections = [(0, 1), (2, 3), (4, 5), (6, 7)]

        no_connections = []

        params = [
            {
                "Persistence__max_alpha_square": [0.5, 0.9, 1.5],
                "Persistence__complex_": ['alpha'],
                "Extract__selected_keypoints": [self.selected_keypoints],
            },
        ]

        return GridSearchCV(pipe, params, n_jobs=1)
