import numpy as np

from ..util import COCOKeypoints


class Person:
    """Represents a person by their keypoints.

    Essentially encapsulates a number of functions on the keypoints that
    the detectors identify as a person.

    Parameters
    ----------
    keypoints : array-like
        shape = [n_keypoints, 3]
    track_index : int, optional
        The index of the track that the person belongs to, if any.

    """
    only_track_arms = False

    def __init__(self, keypoints, track_index=-1):
        self.keypoints = keypoints
        self.track_index = track_index

    def get_nonzero_keypoint(self):
        """Gets the first keypoint from the person that isn't zero.

        Returns
        -------
        keypoint : array-like
            The first non-zero keypoint of the person, shape = [2]

        """
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

    def distance(self, other_person):
        """Calculates the distance between this person and a given person.

        Disregards keypoints that are zero (i.e. not identified by the detector)
        in any of the persons.  If there are no keypoints that are not
        zero for both of the persons, a distance of 10000000 is returned.
        Would have been np.inf, but it does not play nice with scipy.optimize.

        Parameters
        ----------
        other_person : Person object
            The other person to calculate distance to.

        """
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
            return 10000000  # np.inf, but np.inf doesn't play nice with scipy.optimize

        # Calculate average distance between the two people
        distance = np.linalg.norm(xy_person - xy_other)
        distance = distance / xy_person.size

        return distance

    def _filter_nonzero(self, first, second):
        first, second = first[np.nonzero(first)], second[np.nonzero(first)]
        first, second = first[np.nonzero(second)], second[np.nonzero(second)]
        return first, second

    def get_keypoint(self, idx):
        """Gets the keypoint of idx
        Parameters
        ----------
        idx : int

        Returns
        -------
        keypoint : array-like shape = [2]

        """
        return self.keypoints[idx]

    def __getitem__(self, idx):
        return self.keypoints[idx]

    def get_head(self):
        """Gets the keypoint which corresponds to the head of a person

        Returns
        -------
        keypoint : array-like, shape = [3]

        """
        return self.keypoints[COCOKeypoints.Neck.value]

    def get_arm_keypoints(self):
        """Gets keypoints which corresponds to the arms of the Person

        Returns
        -------
        keypoints : array-like, shape = [2, 3]

        """
        return self.keypoints[[COCOKeypoints.RWrist.value, COCOKeypoints.RElbow.value]]

    def is_relevant(self):
        """Checks if the Person has valid keypoints.

        Returns True unless we're only tracking arms, in which case it checks
        if any of the arms have non-zero values.

        Returns
        -------
        bool : boolean

        """
        if Person.only_track_arms:
            return np.any(self.get_arm_keypoints())
        else:
            return True

    def fill_missing_keypoints(self, other):
        """Fills missing keypoints with values from other.

        Parameters
        ----------
        other : Person, used to fill in keypoints which are unidentified
            for this person.

        """
        for i, k in enumerate(self.keypoints):
            if not np.any(k):
                self.keypoints[i] = other.keypoints[i]

    def interpolate(self, other, steps=1):
        """Returns possible interpolated points between the self and other.

        Parameters
        ----------
        other : Person, other person to interpolate positions to.
        steps : int, optional
            The number of interpolated points to return.

        Returns
        -------
        points : array-like
            shape = [steps, Person]

        """
        diff = self.keypoints - other.keypoints
        step_diff = diff / steps
        return [Person(other.keypoints + step_diff * i, self.track_index) for i in range(1, steps)]
