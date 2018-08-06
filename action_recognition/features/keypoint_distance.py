import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class KeypointDistance(BaseEstimator, TransformerMixin):

    def __init__(self, keypoint_pairs):
        self.keypoint_pairs = keypoint_pairs

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, chunks):
        data = np.array([self._keypoint_distance(chunk) for chunk in chunks])

        return data

    def _keypoint_distance(self, chunk):
        distances = np.zeros(len(self.keypoint_pairs))
        for i, (from_, to) in enumerate(self.keypoint_pairs):
            distance = np.linalg.norm(chunk[:, from_, :2] - chunk[:, to, :2])
            average_distance = distance / chunk.shape[0]
            distances[i] = average_distance

        return distances
