"""
Classifiers
===========
Classifiers for action recognition.  TDAClassifier just uses TDA to predict
classes of the data, while EnsembleClassifier combines the TDA approach
with feature engineering to produce more accurate predictions.
"""
from .tda_classifier import TDAClassifier
from .ensemble_classifier import EnsembleClassifier
from .classification_visualiser import ClassificationVisualiser
from .tda_clusterer import TDAClusterer
from .feature_engineering_classifier import FeatureEngineeringClassifier
