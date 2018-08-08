import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TranslateChunksByKeypoints(BaseEstimator, TransformerMixin):
    """Moves the center of each keypoint in each chunk to origin.

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
        """Transforms chunks, copies chunks.

        Parameters
        ----------
        chunks : array-like
            shape = [n_chunks, frames_per_chunk, n_keypoints, x]

        Returns
        -------
        translated_chunks : array-like
            shape = [n_chunks, frames_per_chunk, n_keypoints, x]


        """
        translated_chunks = np.copy(chunks)

        for chunk in translated_chunks:
            self._translate_by_keypoint(chunk)

        return translated_chunks

    def _translate_by_keypoint(self, chunk):
        for i in range(chunk.shape[1]):
            # Don't take unidentified keypoints into account:
            keypoints = chunk[:, i][~np.all(chunk[:, i] == 0, axis=1)]
            if keypoints.shape[0] != 0:
                keypoint_mean = keypoints.mean(axis=0)
                chunk[:, i][~np.all(chunk[:, i] == 0, axis=1)] -= keypoint_mean
