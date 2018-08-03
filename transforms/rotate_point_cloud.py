import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RotatePointCloud(BaseEstimator, TransformerMixin):

    def __init__(self, number_of_added_points=2):
        self.number_of_added_points = number_of_added_points

    def fit(self, X=None, y=None, **fit_params):
        return self

    def transform(self, chunks):
        data = np.array([self._random_rotation(chunk)
                         for chunk in chunks
                         for i in range(self.number_of_added_points)])
        return np.append(data, chunks, axis=0)

    def _random_rotation(self, chunk):
        rotated_chunk = np.copy(chunk)
        rotation_matrix = self._rotation_matrix()

        rotated_chunk = rotated_chunk @ rotation_matrix

        return rotated_chunk

    def _rotation_matrix(self):
        tx = np.radians(np.random.random_integers(low=-180, high=180))
        ty = np.radians(np.random.random_integers(low=-180, high=180))
        tz = np.radians(np.random.random_integers(low=-180, high=180))

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(tx), -np.sin(tx)],
            [0, np.sin(tx), np.cos(tx)]
        ])
        Ry = np.array([
            [np.cos(ty), 0, -np.sin(ty)],
            [0, 1, 0],
            [np.sin(ty), 0, np.cos(ty)]
        ])
        Rz = np.array([
            [np.cos(tz), -np.sin(tz), 0],
            [np.sin(tz), np.cos(tz), 0],
            [0, 0, 1]
        ])

        return Rz
