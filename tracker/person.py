import numpy as np


class Person(object):

    def __init__(self, keypoints, path_index=-1):
        self.keypoints = keypoints
        self.path_index = path_index

    def get_nonzero_keypoint(self):
        return next((keypoint[:2]
                     for keypoint in self.keypoints
                     if not np.array_equal(keypoint[:2], [0.0, 0.0])),
                    [0.0, 0.0])

    def __eq__(self, other_person):
        return np.array_equal(self.keypoints, other_person.keypoints)

    # A person is a [#keypoints x 3] numpy array
    # With [X, Y, Confidence] as the values.
    def distance(self, other_person):
        # Disregard the confidence for now.
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

    def get_head(self):
        return self.keypoints[0]

    def get_hand(self):
        return self.keypoints[4]

    # Result for BODY_25 (25 body parts consisting of COCO + foot)
    # const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
    #     {0,  "Nose"},
    #     {1,  "Neck"},
    #     {2,  "RShoulder"},
    #     {3,  "RElbow"},
    #     {4,  "RWrist"},
    #     {5,  "LShoulder"},
    #     {6,  "LElbow"},
    #     {7,  "LWrist"},
    #     {8,  "MidHip"},
    #     {9,  "RHip"},
    #     {10, "RKnee"},
    #     {11, "RAnkle"},
    #     {12, "LHip"},
    #     {13, "LKnee"},
    #     {14, "LAnkle"},
    #     {15, "REye"},
    #     {16, "LEye"},
    #     {17, "REar"},
    #     {18, "LEar"},
    #     {19, "LBigToe"},
    #     {20, "LSmallToe"},
    #     {21, "LHeel"},
    #     {22, "RBigToe"},
    #     {23, "RSmallToe"},
    #     {24, "RHeel"},
    #     {25, "Background"}
    # };
