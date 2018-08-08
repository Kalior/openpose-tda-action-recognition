import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class KeypointDistance(BaseEstimator, TransformerMixin):
    """Measures the average distance between the given pairs of keypoints.

    Parameters
    ----------
    keypoint_pairs : array of tuples
        Each tuple must contain a from and to index.

    """

    def __init__(self, keypoint_pairs):
        self.keypoint_pairs = keypoint_pairs

    def fit(self, X, y=None, **fit_params):
        """Returns self, as there are no parameters to fit.

        Parameters
        ----------
        X : ignored
        y : ignored
        fit_params : ignored

        Returns
        -------
        self : unchanged

        """
        return self

    def transform(self, chunks):
        """Extracts the average distance between given keypoints.

        Parameters
        ----------
        chunks : array-like
            shape = [n_chunks, frames_per_chunk, n_keypoints, 3]

        Returns
        -------
        data : array-like
            shape = [n_chunks, len(keypoint_pairs)]

        """
        data = np.array([self._keypoint_distance(chunk) for chunk in chunks])

        return data

    def _keypoint_distance(self, chunk):
        distances = np.zeros(len(self.keypoint_pairs))
        for i, (from_, to) in enumerate(self.keypoint_pairs):
            distance = np.linalg.norm(chunk[:, from_, :2] - chunk[:, to, :2])
            average_distance = distance / chunk.shape[0]
            distances[i] = average_distance

        return distances
