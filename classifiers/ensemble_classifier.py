from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

import sklearn_tda as tda
import numpy as np

from .tda_classifier import TDAClassifier
from transforms import AverageSpeed, AngleChangeSpeed, AmountOfMovement,\
    TranslateChunks, SmoothChunks, FlattenTo3D, Persistence, ExtractKeypoints, \
    InterpolateKeypoints, KeypointDistance
from util import COCOKeypoints, coco_connections


class EnsembleClassifier(BaseEstimator, ClassifierMixin):

    def fit(self, X, y, **fit_params):
        sliced_wasserstein_classifier = TDAClassifier(cross_validate=False)

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

        arm_keypoints = [
            COCOKeypoints.RShoulder.value,
            COCOKeypoints.LShoulder.value,
            COCOKeypoints.RElbow.value,
            COCOKeypoints.LElbow.value,
            COCOKeypoints.RWrist.value,
            COCOKeypoints.LWrist.value
        ]
        arm_connections = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (4, 5)]

        other_tda = Pipeline([
            ("Translate",   TranslateChunks()),
            ("Smoothing",   SmoothChunks()),
            ("Extract",     ExtractKeypoints(arm_keypoints)),
            ("Interpolate", InterpolateKeypoints(arm_connections)),
            ("Flattening",  FlattenTo3D()),
            ("Persistence", Persistence()),
            ("Separator", tda.DiagramSelector(limit=np.inf, point_type="finite")),
            ("Union", FeatureUnion([
                ("PersistenceImage", persistence_image),
                ("Landscape", landscape),
                ("TopologicalVector", topological_vector),
                ("Silhouette", silhouette),
                ("BettiCurve", betti_curve)
            ]))
        ])

        connections = [
            (COCOKeypoints.RWrist.value, COCOKeypoints.LWrist.value),
            (COCOKeypoints.RElbow.value, COCOKeypoints.LElbow.value),
            (COCOKeypoints.Neck.value, COCOKeypoints.LAnkle.value),
            (COCOKeypoints.Neck.value, COCOKeypoints.RAnkle.value),
            (COCOKeypoints.LWrist.value, COCOKeypoints.LAnkle.value),
            (COCOKeypoints.RWrist.value, COCOKeypoints.RAnkle.value)
        ]

        feature_union = FeatureUnion([
            ("AverageSpeed", AverageSpeed(range(18))),
            ("AngleChangeSpeed", AngleChangeSpeed(coco_connections)),
            ("Movement",    AmountOfMovement(range(18))),
            ("KeypointDistance", KeypointDistance(connections)),
            ("TDA", other_tda)
        ])
        feature_union_pipeline = Pipeline([
            ("Union", feature_union),
            ("Estimator", SVC(probability=True))
        ])

        persistence_scale_space = Pipeline([
            ("Translate",   TranslateChunks()),
            ("Smoothing",   SmoothChunks()),
            ("Extract",     ExtractKeypoints(arm_keypoints)),
            ("Interpolate", InterpolateKeypoints(arm_connections)),
            ("Flattening",  FlattenTo3D()),
            ("Persistence", Persistence()),
            ("Separator",   tda.DiagramSelector(limit=np.inf, point_type="finite")),
            ("TDA",         tda.PersistenceScaleSpace()),
            ("Estimator",   SVC(kernel='precomputed', probability=True))
        ])

        persistence_weighted_gaussian = Pipeline([
            ("Translate",   TranslateChunks()),
            ("Smoothing",   SmoothChunks()),
            ("Extract",     ExtractKeypoints(arm_keypoints)),
            ("Interpolate", InterpolateKeypoints(arm_connections)),
            ("Flattening",  FlattenTo3D()),
            ("Persistence", Persistence()),
            ("Separator",   tda.DiagramSelector(limit=np.inf, point_type="finite")),
            ("TDA",         tda.PersistenceWeightedGaussian()),
            ("Estimator",   SVC(kernel='precomputed', probability=True))
        ])
        # classifier = feature_union_pipeline

        # Can't use multiple jobs because lambda in persistence image isn't pickable
        classifier = VotingClassifier(estimators=[
            ("Union", feature_union_pipeline),
            ("SWKernel", sliced_wasserstein_classifier),
            ("PWGKernel", persistence_weighted_gaussian)
            # ("PSSKernel", persistence_scale_space)
        ], voting='soft', n_jobs=1)

        self.classifier = classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)
