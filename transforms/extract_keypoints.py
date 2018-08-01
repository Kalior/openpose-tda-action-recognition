import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ExtractKeypoints(BaseEstimator, TransformerMixin):

    def __init__(self, selected_keypoints=range(18)):
        self.selected_keypoints = selected_keypoints

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, chunks):
        return np.array([chunk[:, self.selected_keypoints, :2] for chunk in chunks])
