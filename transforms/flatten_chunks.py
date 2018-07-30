import numpy as np


class Flatten:

    def __init__(self, selected_keypoints):
        self.selected_keypoints = selected_keypoints

    def transform(self, chunks):
        data = np.array([chunk[:, self.selected_keypoints, :2].flatten()
                         for chunk in chunks])

        return data
