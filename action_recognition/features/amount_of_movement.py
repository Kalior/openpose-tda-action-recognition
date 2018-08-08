import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AmountOfMovement(BaseEstimator, TransformerMixin):
    """Measures the absolute distance of each chunk between the first and last frame.

    Parameters
    ----------
    selected_keypoints : array-like
        Indicies for which the absolute movement should be measured.

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
        """Extracts the absolute movement from the chunks.

        Parameters
        ----------
        chunks : array-like
            shape = [n_chunks, frames_per_chunk, n_keypoints, 3]

        Returns
        -------
        data : array-like
            shape = [n_chunks, 1]

        """
        data = np.array([self._movement(chunk)
                         for chunk in chunks])
        return data

    def _movement(self, chunk):
        start_point = chunk[0, self.selected_keypoints, :2]
        end_point = chunk[-1, self.selected_keypoints, :2]
        distance = np.linalg.norm(end_point - start_point)
        return np.array([distance])
