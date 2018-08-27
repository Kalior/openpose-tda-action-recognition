import numpy as np

from ..util import COCOKeypoints, coco_connections


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

    def __init__(self, keypoints, track_index=-1):
        self.keypoints = keypoints
        self.track_index = track_index
        self.og_keypoints = np.copy(keypoints)

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

    def fill_missing_keypoints(self, other, fill_type='copy'):
        """Fills missing keypoints.

        Either fills so that the difference between keypoints is consistent with other,
        or copies the keypoint position in other.

        Parameters
        ----------
        other : Person, used to fill in keypoints which are unidentified
            for this person.
        fill_type : str, optional, default = 'copy'
            Either 'copy' or 'diff', as explained above.

        """
        # If type is diff, we have to try to fill in both directions.
        if fill_type is 'diff':
            self._fill_diff_loop(enumerate(self.keypoints), other)
            self._fill_diff_loop(reversed(list(enumerate(self.keypoints))), other)

        # Always fill with copy in case we did not find any possible
        # connections.
        for i, k in enumerate(self.keypoints):
            if not np.any(k):
                self.keypoints[i] = np.copy(other.keypoints[i])

    def _fill_diff_loop(self, enumerator, other):
        for i, k in enumerator:
            if not np.any(k):
                new_keypoint = self._diff_fill_keypoint(i, other)
                if new_keypoint is not None:
                    self.keypoints[i] = new_keypoint

    def _diff_fill_keypoint(self, keypoint_index, other):
        # Try finding a keypoint in the downwards direction (e.g. shoulder - elbow)
        connect_downwards = next((from_ for from_, to in coco_connections
                                  if to == keypoint_index and
                                  np.any(other.keypoints[from_]) and
                                  np.any(other.keypoints[to]) and
                                  np.any(self.keypoints[from_])
                                  ), -1)
        # If none found, try the other direction. (e.g. elbow - shoulder)
        # This may require a reversed iteration through the keypoints as well,
        # to fill every keypoint.
        connect_upwards = next((to for from_, to in coco_connections
                                if from_ == keypoint_index and
                                np.any(other.keypoints[to]) and
                                np.any(other.keypoints[from_]) and
                                np.any(self.keypoints[to])
                                ), -1)
        if connect_downwards == -1 and connect_upwards == -1:
            return None

        # Take the first index which isn't -1.
        connect_i = [i for i in [connect_downwards, connect_upwards] if i != -1][0]

        diff = other.keypoints[connect_i] - other.keypoints[keypoint_index]
        return self.keypoints[connect_i] - diff

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

    def reset_keypoints(self):
        """Resets keypoints to the original copy saved at creation.
        """
        self.keypoints = np.copy(self.og_keypoints)
