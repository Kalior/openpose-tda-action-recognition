import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TranslateChunks(BaseEstimator, TransformerMixin):
    """Moves the center of each chunk to origin.

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
            self._normalise_chunk(chunk)

        return translated_chunks

    def _normalise_chunk(self, chunk):
        # Don't take unidentified keypoints into account:
        mean = chunk[~np.all(chunk == 0, axis=2)].mean(axis=0)

        chunk[~np.all(chunk == 0, axis=2)] -= mean
