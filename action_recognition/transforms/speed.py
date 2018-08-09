import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Speed(BaseEstimator, TransformerMixin):
    """Transforms chunks into the speed with which each keypoint moves.

    Parameters
    ----------
    window : int, optional
        The number of frames to take into account when calculating speed.

    """

    def __init__(self, window=5):
        self.window = window

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
        """Transforms chunks

        Parameters
        ----------
        chunks : array-like
            shape = [n_chunks, frames_per_chunk, n_keypoints, x]

        Returns
        -------
        data : array-like
            shape = [n_chunks, frames_per_chunk - window, n_keypoints, x]


        """

        data = np.array([self._chunk_to_speed(chunk) for chunk in chunks])

        return data

    def _chunk_to_speed(self, chunk):
        number_of_frames = chunk.shape[0] - self.window
        speed_chunk = np.zeros((number_of_frames, *chunk.shape[1:]))

        for i in range(number_of_frames):
            end_index = self.window + i
            chunk_window = chunk[i:end_index, :, :]
            # Remove last index as it corresponds |last - first|
            pairwise_distance = np.abs(chunk_window - np.roll(chunk_window, 1, axis=0))[:-1]
            total_distance = pairwise_distance.sum(axis=0)
            speed_chunk[i] = total_distance / self.window

        return speed_chunk
