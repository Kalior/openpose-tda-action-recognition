import numpy as np


class Person(object):

    def __init__(self, path_index, keypoints):
        self.path_index = path_index
        self.keypoints = keypoints

    def get_nonzero_keypoint(self):
        return next((keypoint[:2]
                     for keypoint in self.keypoints
                     if not np.array_equal(keypoint[:2], [0.0, 0.0])),
                    [0.0, 0.0])
