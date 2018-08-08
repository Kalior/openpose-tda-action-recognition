import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class InterpolateKeypoints(BaseEstimator, TransformerMixin):
    """Interpolates between the specified keypoints.

    I.e., creates intermediate points between connect_keypoints which
    can help enrichen the point clouds inputted into the Persistence
    calculator.

    Parameters
    ----------
    connect_keypoints : array of tuples
        Each tuple should have the form (from, to) which specifies which
        points in the data to add points between.
    number_of_points : int, optional, default=2
        Specifies how many points to add between every pair of points.

    """

    def __init__(self, connect_keypoints, number_of_points=2):
        self.connect_keypoints = connect_keypoints
        self.number_of_points = number_of_points

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
        """Adds interpolated points between the specified keypoints to the input.

        Parameters
        ----------
        chunks : array-like
            shape = [n_chunks, frames_per_chunk, n_keypoints, 2]
            Adds values to each subarray in the array.

        Returns
        -------
        chunks : array-like
            shape = [n_chunks, frames_per_chunk,
                n_keypoints + len(connect_keypoints) * number_of_points
                , 2]

        """
        return np.array([self._connect_keypoints(chunk) for chunk in chunks])

    def _connect_keypoints(self, chunk):
        new_number_of_keypoints = chunk.shape[1] + \
            len(self.connect_keypoints) * self.number_of_points

        connected_chunk = np.zeros((chunk.shape[0], new_number_of_keypoints, chunk.shape[2]))
        connected_chunk[:, :chunk.shape[1]] = chunk

        for i, frame in enumerate(chunk):
            for j, (from_, to) in enumerate(self.connect_keypoints):
                start_index = j * self.number_of_points + chunk.shape[1]
                intermediate_points = self._intermediate_points(
                    frame, from_, to, self.number_of_points)
                for k, points in enumerate(intermediate_points):
                    connected_chunk[i, start_index + k] = points

        return connected_chunk

    def _intermediate_points(self, frame, from_, to, number_of_points):
        from_point = frame[from_]
        to_point = frame[to]
        diff = from_point - to_point
        step = diff / (number_of_points + 1)
        step_array = np.tile(step, number_of_points).reshape(-1, step.shape[0])

        intermediate_points = to_point + step_array * \
            np.arange(1, number_of_points + 1)[:, np.newaxis]
        return intermediate_points
