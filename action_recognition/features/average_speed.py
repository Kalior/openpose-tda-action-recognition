import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AverageSpeed(BaseEstimator, TransformerMixin):
    """Measures the average speed with which the given keypoints moves.

    Parameters
    ----------
    selected_keypoints : array-like
        An list with indicies for the keypoints to measure the movement of.

    """

    def __init__(self, selected_keypoints):
        self.selected_keypoints = selected_keypoints

    def fit(self, X=None, y=None, **fit_params):
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
        """Extracts the average speed with which the given keypoints changes.

        Parameters
        ----------
        chunks : array-like
            shape = [n_chunks, frames_per_chunk, n_keypoints, 3]

        Returns
        -------
        data : array-like
            shape = [n_chunks, len(selected_keypoints)]

        """
        data = np.array([self._speed_of_chunk(chunk)
                         for chunk in chunks])

        return data

    def _speed_of_chunk(self, chunk):
        speed_chunk = np.zeros((len(self.selected_keypoints),))
        for i, k in enumerate(self.selected_keypoints):
            part = chunk[:, k, :2]
            # Remove last value as it corresponds to last value - first value
            pairwise_distance = np.abs(part - np.roll(part, 1, axis=0))[:-1]
            total_distance = pairwise_distance.sum(axis=0)
            speed_chunk[i] = np.linalg.norm(total_distance / chunk.shape[0])

        return speed_chunk
