from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import RobustScaler

import sklearn_tda as tda
import numpy as np

from .tda_classifier import TDAClassifier
from ..transforms import TranslateChunks, SmoothChunks, FlattenTo3D, Persistence, \
    ExtractKeypoints, InterpolateKeypoints
from ..features import AverageSpeed, AngleChangeSpeed, AmountOfMovement, KeypointDistance
from ..util import COCOKeypoints, coco_connections


class FeatureEngineeringClassifier(BaseEstimator, ClassifierMixin):
    """Classifier for actions.

    Makes use of features extracted from the data using other vectorisations
    from sklearn_tda, and from the features module.

    Parameters
    ----------
    use_tda_vectorisations : boolean, optional, default = False
        Determines if the vectorisations from the sklearn_tda library
        should be used. *Note*: Can only use one thread, since some
        parts of sklearn_tda vectorisations are not pickable. This also
        means that the model can't be saved to disk.
    """

    def __init__(self, use_tda_vectorisations=False):
        self.use_tda_vectorisations = use_tda_vectorisations

        self.keypoint_distance_connections = [(k1.value, k2.value) for k1, k2 in [
            (COCOKeypoints.RWrist, COCOKeypoints.LWrist),
            (COCOKeypoints.RElbow, COCOKeypoints.LElbow),
            (COCOKeypoints.Neck, COCOKeypoints.LAnkle),
            (COCOKeypoints.Neck, COCOKeypoints.RAnkle),
            (COCOKeypoints.LWrist, COCOKeypoints.LAnkle),
            (COCOKeypoints.RWrist, COCOKeypoints.RAnkle)
        ]]

        self.angle_change_connections = np.array(coco_connections)
        self.speed_keypoints = range(14)
        self.arm_keypoints = [k.value for k in [
            COCOKeypoints.RElbow,
            COCOKeypoints.RWrist,
            COCOKeypoints.LElbow,
            COCOKeypoints.LWrist,
            COCOKeypoints.LKnee,
            COCOKeypoints.LAnkle,
            COCOKeypoints.RKnee,
            COCOKeypoints.RAnkle
        ]]
        self.arm_connections = [(0, 1), (2, 3), (4, 5), (6, 7)]

    def fit(self, X, y, **fit_params):
        """Fit the model.

        Parameters
        ----------
        X : iterable
            Training data
        y : iterable
            Training labels.
        fit_params : dict
            ignored for now.

        Returns
        -------
        self

        """

        classifier = Pipeline([
            ("Union", self._feature_engineering_union()),
            ("Estimator", SVC(probability=True))
        ])

        self.classifier = classifier.fit(X, y)
        return self

    def predict(self, X):
        """Predicts using the pipeline.

        Parameters
        ----------
        X : iterable
            Data to predict labels for.

        Returns
        -------
        y_pred : array-like

        """
        return self.classifier.predict(X)

    def predict_proba(self, X):
        """Predicts using the pipeline.

        Parameters
        ----------
        X : iterable
            Data to predict labels for.

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]

        """
        return self.classifier.predict_proba(X)

    def _feature_engineering_union(self):
        transformer_list = [
            ("AverageSpeed", Pipeline([
                ("Feature", AverageSpeed(self.speed_keypoints)),
                ("Scaler", RobustScaler())
            ])),
            ("AngleChangeSpeed", Pipeline([
                ("Feature", AngleChangeSpeed(self.angle_change_connections)),
                ("Scaler", RobustScaler())
            ])),
            ("Movement",    Pipeline([
                ("Feature", AmountOfMovement(range(14))),
                ("Scaler", RobustScaler())
            ])),
            ("KeypointDistance", Pipeline([
                ("Feature", KeypointDistance(self.keypoint_distance_connections)),
                ("Scaler", RobustScaler())
            ]))
        ]
        if self.use_tda_vectorisations:
            transformer_list.append(("TDAVectorisations", self._tda_vectorisations_pipeline()))

        return FeatureUnion(transformer_list)

    def _tda_vectorisations_pipeline(self):
        persistence_image = Pipeline([
            ("Rotator", tda.DiagramPreprocessor(scaler=tda.BirthPersistenceTransform())),
            ("PersistenceImage", tda.PersistenceImage()),
            ("Scaler", RobustScaler())
        ])

        return Pipeline([
            ("Translate",   TranslateChunks()),
            ("Extract",     ExtractKeypoints(self.arm_keypoints)),
            ("Smoothing",   SmoothChunks()),
            ("Interpolate", InterpolateKeypoints(self.arm_connections)),
            ("Flattening",  FlattenTo3D()),
            ("Persistence", Persistence()),
            ("Separator", tda.DiagramSelector(limit=np.inf, point_type="finite")),
            ("Union", FeatureUnion([
                ("PersistenceImage", persistence_image),
                ("Landscape", Pipeline([
                    ("TDA", tda.Landscape(resolution=10)),
                    ("Scaler", RobustScaler())
                ])),
                ("TopologicalVector", Pipeline([
                    ("TDA", tda.TopologicalVector()),
                    ("Scaler", RobustScaler())
                ])),
                ("Silhouette", Pipeline([
                    ("TDA", tda.Silhouette()),
                    ("Scaler", RobustScaler())
                ])),
                ("BettiCurve", Pipeline([
                    ("TDA", tda.BettiCurve()),
                    ("Scaler", RobustScaler())
                ]))
            ])),
            ("Scaler", RobustScaler())
        ])
