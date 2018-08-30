import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FlattenTo3D(BaseEstimator, TransformerMixin):
    """Transforms chunks into 3D point clouds.

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
        """Transforms chunks into a 3D point cloud.

        Parameters
        ----------
        chunks : array-like, shape = [n_chunks, frames_per_chunk, n_keypoints, 2]
            each subarray in chunks is transformed from
            [frames_per_chunk, n_keypoints, 2] into [frames_per_chunk * n_keypoints, 3]

        Returns
        -------
        data : array-like, shape = [n_chunks, frames_per_chunk * n_keypoints, 3]
            array of point clouds.

        """
        data = np.array([self._chunk_to_3D(chunk) for chunk in chunks])

        return data

    def _chunk_to_3D(self, chunk):
        number_of_frames = chunk.shape[0]

        flat_chunk = np.vstack(chunk)

        third_dimension = np.repeat(np.arange(0, number_of_frames), chunk.shape[1])

        return np.c_[flat_chunk, third_dimension]
