import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AngleChangeSpeed(BaseEstimator, TransformerMixin):

    def __init__(self, connect_keypoints):
        self.connect_keypoints = connect_keypoints

    def fit(self, X=None, y=None, **fit_params):
        return self

    def transform(self, chunks):
        data = np.array([self._angle_change_speed(chunk)
                         for chunk in chunks])
        return data

    def _angle_change_speed(self, chunk):
        angle_change = np.zeros((len(self.connect_keypoints),))

        for i, (from_, to) in enumerate(self.connect_keypoints):
            from_part = chunk[:, from_, :2]
            to_part = chunk[:, to, :2]
            angles = self._angles_between_arrays(from_part, to_part)

            # Remove last value as it corresponds to last value - first value
            pairwise_change = np.abs(angles - np.roll(angles, 1, axis=0))[:-1]
            total_change = pairwise_change.sum(axis=0)
            angle_change[i] = np.linalg.norm(total_change / chunk.shape[0])

        return angle_change

    def _angles_between_arrays(self, from_part, to_part):
        angles = np.zeros(len(from_part))
        for i in range(len(from_part)):
            if np.linalg.norm(from_part[i]) == 0 or np.linalg.norm(to_part[i]) == 0:
                angles[i] = 0
            else:
                v_from = from_part[i] / np.linalg.norm(from_part[i])
                v_to = to_part[i] / np.linalg.norm(to_part[i])
                angles[i] = np.arccos(np.clip(np.dot(v_from, v_to), -1.0, 1.0))
        return angles
