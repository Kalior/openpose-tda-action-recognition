import cv2
import numpy as np

from util import COCOKeypoints


class PathVisualiser(object):

    def __init__(self):
        self.colors = [(255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]

    def draw_paths(self, people_paths, img, current_frame, only_track_arms=False):
        for i, person_path in enumerate(people_paths):
            self._add_lines_from_path(img, person_path,
                                      self.colors[i % len(self.colors)], current_frame, only_track_arms)

        cv2.imshow("output", img)
        cv2.waitKey(15)

    def _add_lines_from_path(self, img, person_path, color, current_frame, only_track_arms):
        # Don't draw old paths
        if person_path.last_frame_update <= current_frame - 10:
            return

        if only_track_arms:
            path = person_path.get_keypoint_path(COCOKeypoints.RWrist.value)
        else:
            path = person_path.get_keypoint_path(COCOKeypoints.Neck.value)

        start_index = max(1, len(path) - 10)
        for i in range(start_index, len(path)):
            keypoint = path[i].astype(np.int)
            prev_keypoint = path[i - 1].astype(np.int)

            cv2.line(img, tuple(prev_keypoint), tuple(keypoint), color, 3)
