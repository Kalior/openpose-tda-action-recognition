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


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Classifier for actions.

    Makes use of the tda_classifier and combines this with
    features extracted from the data using other vectorisations
    from sklearn_tda, and from the features module.
    """

    def __init__(self):
        self.keypoint_distance_connections = [(k1.value, k2.value) for k1, k2 in [
            (COCOKeypoints.RWrist, COCOKeypoints.LWrist),
            (COCOKeypoints.RElbow, COCOKeypoints.LElbow),
            (COCOKeypoints.Neck, COCOKeypoints.LAnkle),
            (COCOKeypoints.Neck, COCOKeypoints.RAnkle),
            (COCOKeypoints.LWrist, COCOKeypoints.LAnkle),
            (COCOKeypoints.RWrist, COCOKeypoints.RAnkle)
        ]]

        self.angle_change_connections = np.array(coco_connections)
        self.speed_keypoints = range(18)
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
            Training data, must be passable to transforms.TranslateChunks()
        y : iterable
            Training labels.
        fit_params : dict
            ignored for now.

        Returns
        -------
        self

        """
        sliced_wasserstein_classifier = TDAClassifier(cross_validate=False)

        feature_union_pipeline = Pipeline([
            ("Union", self._feature_engineering_union()),
            ("Estimator", SVC(probability=True))
        ])

        # Can't use multiple jobs because lambda in persistence image isn't pickable
        classifier = VotingClassifier(estimators=[
            ("Union", feature_union_pipeline),
            ("SWKernel", sliced_wasserstein_classifier)
        ], voting='soft', n_jobs=1)

        self.classifier = classifier.fit(X, y)
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
        return self.classifier.predict(X)

    def _tda_vectorisations_pipeline(self):
        persistence_image = Pipeline([
            ("Rotator", tda.DiagramPreprocessor(scaler=tda.BirthPersistenceTransform())),
            ("PersistenceImage", tda.PersistenceImage())
        ])

        landscape = Pipeline([
            ("Rotator", tda.DiagramPreprocessor(use=False, scaler=tda.BirthPersistenceTransform())),
            ("Landscape", tda.Landscape(resolution=1000))
        ])

        topological_vector = Pipeline([
            ("Rotator", tda.DiagramPreprocessor(use=False, scaler=tda.BirthPersistenceTransform())),
            ("TopologicalVector", tda.TopologicalVector())
        ])

        silhouette = Pipeline([
            ("Rotator", tda.DiagramPreprocessor(use=False, scaler=tda.BirthPersistenceTransform())),
            ("Silhouette", tda.Silhouette())
        ])

        betti_curve = Pipeline([
            ("Rotator", tda.DiagramPreprocessor(use=False, scaler=tda.BirthPersistenceTransform())),
            ("BettiCurve", tda.BettiCurve())
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
                ("Landscape", landscape),
                ("TopologicalVector", topological_vector),
                ("Silhouette", silhouette),
                ("BettiCurve", betti_curve)
            ])),
            ("Scaler", RobustScaler())
        ])

    def _feature_engineering_union(self):
        return FeatureUnion([
            ("AverageSpeed", Pipeline([
                ("Feature", AverageSpeed(self.speed_keypoints)),
                ("Scaler", RobustScaler())
            ])),
            ("AngleChangeSpeed", Pipeline([
                ("Feature", AngleChangeSpeed(self.angle_change_connections)),
                ("Scaler", RobustScaler())
            ])),
            ("Movement",    Pipeline([
                ("Feature", AmountOfMovement(range(18))),
                ("Scaler", RobustScaler())
            ])),
            ("KeypointDistance", Pipeline([
                ("Feature", KeypointDistance(self.keypoint_distance_connections)),
                ("Scaler", RobustScaler())
            ]))
        ])
