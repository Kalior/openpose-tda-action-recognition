import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Flatten(BaseEstimator, TransformerMixin):
    """Flattens the chunks passed to transform.

    """

    def fit(self, X, y=None, **fit_params):
        """Returns self unchanged, as there are no parameters to fit.

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
        """Flattens each subarray in chunks.

        Parameters
        ----------
        chunks : array-like
            array to flatten, shape = [n_chunks, n_frames, n_keypoints, x]

        Returns
        -------
        data : array-like
            shape = [n_chunks, n_frames * n_keypoints * x]

        """
        data = np.array([chunk.flatten() for chunk in chunks])

        return data
