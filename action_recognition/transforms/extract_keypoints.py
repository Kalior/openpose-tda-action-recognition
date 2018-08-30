import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ExtractKeypoints(BaseEstimator, TransformerMixin):
    """Extracts a the specified keypoints from the inputted chunks.

    Will also remove the confidence from each keypoint in the chunk
    (the 3rd value).

    Parameters
    ----------
    selected_keypoints : array-like, optional
        Contains the indicies of the keypoints to extract.
    """

    def __init__(self, selected_keypoints=range(18)):
        self.selected_keypoints = selected_keypoints

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
        """Extracts the specified keypoints from the input.

        Parameters
        ----------
        chunks : array-like
            The chunks to extract keypoints from.
            expected shape = [n_chunks, frames_per_chunk, n_keypoints, 3].

        Returns
        -------
        chunks : array-like
            The chunks with the specified keypoints extracted and the last
            value removed. shape = [n_chunks, frames_per_chunk, n_selected_keypoints, 2].

        """
        return np.array([chunk[:, self.selected_keypoints, :2] for chunk in chunks])
