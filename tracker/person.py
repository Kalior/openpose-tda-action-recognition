import numpy as np

from util import COCOKeypoints


class Person(object):
    only_track_arms = False

    def __init__(self, keypoints, path_index=-1):
        self.keypoints = keypoints
        self.path_index = path_index

    def get_nonzero_keypoint(self):
        if Person.only_track_arms:
            relevant_keypoints = self.get_arm_keypoints()
        else:
            relevant_keypoints = self.keypoints

        return next((keypoint[:2]
                     for keypoint in relevant_keypoints
                     if np.any(keypoint[:2])),
                    np.array([0.0, 0.0]))

    def __eq__(self, other_person):
        return np.array_equal(self.keypoints, other_person.keypoints)

    # A person is a [#keypoints x 3] numpy array
    # With [X, Y, Confidence] as the values.
    def distance(self, other_person):
        # Disregard the confidence for now.
        if Person.only_track_arms:
            xy_person = self.get_arm_keypoints()[:, :2]
            xy_other = other_person.get_arm_keypoints()[:, :2]
        else:
            xy_person = self.keypoints[:, :2]
            xy_other = other_person.keypoints[:, :2]

        #   Don't include the keypoints we didn't identify
        # as this can give large frame-to-frame errors.
        xy_person, xy_other = self._filter_nonzero(xy_person, xy_other)

        if xy_person.size == 0:
            return 10000  # np.inf, but np.inf doesn't play nice with scipy.optimize

        # Calculate average distance between the two people
        distance = np.linalg.norm(xy_person - xy_other)
        distance = distance / xy_person.size

        return distance

    def _filter_nonzero(self, first, second):
        first, second = first[np.nonzero(first)], second[np.nonzero(first)]
        first, second = first[np.nonzero(second)], second[np.nonzero(second)]
        return first, second

    def get_keypoint(self, idx):
        return self.keypoints[idx]

    def __getitem__(self, idx):
        return self.keypoints[idx]

    def get_head(self):
        return self.keypoints[COCOKeypoints.Neck.value]

    def get_arm_keypoints(self):
        return self.keypoints[[COCOKeypoints.RWrist.value, COCOKeypoints.RElbow.value]]

    def is_relevant(self):
        if Person.only_track_arms:
            return np.any(self.get_arm_keypoints())
        else:
            return True
