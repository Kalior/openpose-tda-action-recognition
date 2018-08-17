from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import RobustScaler

import sklearn_tda as tda
import numpy as np

from .tda_classifier import TDAClassifier
from .feature_engineering_classifier import FeatureEngineeringClassifier
from ..transforms import TranslateChunks, SmoothChunks, FlattenTo3D, Persistence, \
    ExtractKeypoints, InterpolateKeypoints
from ..features import AverageSpeed, AngleChangeSpeed, AmountOfMovement, KeypointDistance
from ..util import COCOKeypoints, coco_connections


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Classifier for actions.

    Makes use of the tda_classifier and combines this with
    features extracted from the data using other vectorisations
    from sklearn_tda, and from the features module.

    *Note*: Can only use one thread, since some parts of sklearn_tda vectorisations
    are not pickable.
    """

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

        feature_union_classifier = FeatureEngineeringClassifier()

        # Can't use multiple jobs because lambda in persistence image isn't pickable
        classifier = VotingClassifier(estimators=[
            ("Union", feature_union_classifier),
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
