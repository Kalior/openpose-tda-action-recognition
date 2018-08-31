import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..util import COCOKeypoints


class ArmLegRatio(BaseEstimator, TransformerMixin):
    """Measures the ratio between the length of the arms and legs of a person.

    The idea is that it will be useful since perspective effects should
    have a greater impact on people lying down than people standing up.
    """

    def __init__(self):
        self.arms_indicies = [
            [
                COCOKeypoints.RShoulder.value,
                COCOKeypoints.RElbow.value,
                COCOKeypoints.RWrist.value
            ],
            [
                COCOKeypoints.LShoulder.value,
                COCOKeypoints.LElbow.value,
                COCOKeypoints.LWrist.value
            ]
        ]

        self.legs_indicies = [
            [
                COCOKeypoints.RHip.value,
                COCOKeypoints.RKnee.value,
                COCOKeypoints.RAnkle.value
            ],
            [
                COCOKeypoints.LHip.value,
                COCOKeypoints.LKnee.value,
                COCOKeypoints.LAnkle.value
            ]
        ]

    def fit(self, X=None, y=None, **fit_params):
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
        """Calculate the average arm to leg ratio for the chunks.

        Parameters
        ----------
        chunks : array-like
            shape = [n_chunks, frames_per_chunk, n_keypoints, 3]

        Returns
        -------
        data : array-like
            shape = [n_chunks, 1]

        """
        data = np.array([self._leg_arm_ratio(chunk)
                         for chunk in chunks])
        return data

    def _leg_arm_ratio(self, chunk):
        arm_length = 0
        for arm_index in range(2):
            arm = chunk[:, self.arms_indicies[arm_index], :2]
            arm_length += np.linalg.norm(arm[:, 0, :] - arm[:, 1, :], axis=1)
            arm_length += np.linalg.norm(arm[:, 1, :] - arm[:, 2, :], axis=1)

        leg_length = 0
        for leg_index in range(2):
            leg = chunk[:, self.legs_indicies[leg_index], :2]
            leg_length += np.linalg.norm(leg[:, 0, :] - leg[:, 1, :], axis=1)
            leg_length += np.linalg.norm(leg[:, 1, :] - leg[:, 2, :], axis=1)

        # Don't count values where there is no length
        mask = ((arm_length != 0) & (leg_length != 0))
        ratio = arm_length[mask] / leg_length[mask]
        if len(ratio) > 0:
            average_ratio = ratio.mean()
            return np.array([average_ratio])
        else:
            return np.array([0])
