import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AmountOfMovement(BaseEstimator, TransformerMixin):

    def __init__(self, selected_keypoints):
        self.selected_keypoints = selected_keypoints

    def fit(self, X=None, y=None, **fit_params):
        return self

    def transform(self, chunks):
        data = np.array([self._movement(chunk)
                         for chunk in chunks])
        return data

    def _movement(self, chunk):
        start_point = chunk[0, self.selected_keypoints, :2]
        end_point = chunk[-1, self.selected_keypoints, :2]
        distance = np.linalg.norm(end_point - start_point)
        return np.array([distance])
