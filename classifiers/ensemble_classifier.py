from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

from .tda_classifier import TDAClassifier
from transforms import AverageSpeed, AngleChangeSpeed, AmountOfMovement
from util import coco_connections


class EnsembleClassifier(BaseEstimator, ClassifierMixin):

    def fit(self, X, y, **fit_params):
        tda_classifier = TDAClassifier(cross_validate=False)

        speed_pipeline = Pipeline([
            ("AverageSpeed", AverageSpeed(range(18))),
            ("Estimator",    SVC(probability=True))
        ])

        angle_pipeline = Pipeline([
            ("AngleChangeSpeed", AngleChangeSpeed(coco_connections)),
            ("Estimator",        SVC(probability=True))
        ])

        movement_pipeline = Pipeline([
            ("Movement",    AmountOfMovement(range(18))),
            ("Estimator",   SVC(probability=True))
        ])

        classifier = VotingClassifier(estimators=[
            ("TDA", tda_classifier),
            ("Speed", speed_pipeline),
            ("Angle", angle_pipeline),
            ("Movement", movement_pipeline)
        ], voting='soft', n_jobs=-1)

        self.classifier = classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)
