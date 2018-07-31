import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Flatten(BaseEstimator, TransformerMixin):

    def __init__(self, selected_keypoints):
        self.selected_keypoints = selected_keypoints

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, chunks):
        data = np.array([chunk[:, self.selected_keypoints, :2].flatten()
                         for chunk in chunks])

        return data
