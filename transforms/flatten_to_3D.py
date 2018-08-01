import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FlattenTo3D(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, chunks):
        data = np.array([self._chunk_to_3D(chunk) for chunk in chunks])

        return data

    def _chunk_to_3D(self, chunk):
        number_of_frames = chunk.shape[0]

        flat_chunk = np.vstack(chunk)

        third_dimension = np.repeat(np.arange(0, number_of_frames), chunk.shape[1])

        return np.c_[flat_chunk, third_dimension]
