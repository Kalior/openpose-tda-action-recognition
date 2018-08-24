from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier

import sklearn_tda as tda
import numpy as np

from .tda_classifier import TDAClassifier
from .feature_engineering_classifier import FeatureEngineeringClassifier


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Classifier for actions.

    Makes use of the tda_classifier and combines this with
    features extracted from the data using other vectorisations
    from sklearn_tda, and from the features module.

    *Note*: Can only use one thread, since some parts of sklearn_tda vectorisations
    are not pickable.

    Parameters
    ----------
    use_tda_vecorisations : boolean, optional, default=False
        Specifies if the vectorisations from sklearn_tda should be
        part of the feature_engineering pipeline.
    """

    def __init__(self, use_tda_vectorisations=False):
        self.use_tda_vectorisations = use_tda_vectorisations

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

        feature_union_classifier = FeatureEngineeringClassifier(
            use_tda_vectorisations=self.use_tda_vectorisations)

        # Can't use multiple jobs since the lambdas in some parts of sklearn_tda aren't pickable
        classifier = VotingClassifier(estimators=[
            ("Union", feature_union_classifier),
            ("SWKernel", sliced_wasserstein_classifier)
        ], voting='soft', n_jobs=1)

        self.classifier = classifier.fit(X, y)
        self.classes_ = classifier.classes_
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
