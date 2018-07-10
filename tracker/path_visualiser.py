import cv2
import numpy as np


class PathVisualiser(object):

    def __init__(self):
        self.colors = [(255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]

    def draw_paths(self, people_paths, img, current_frame, only_arms=False):
        for i, person_path in enumerate(people_paths):
            self._add_lines_from_path(img, person_path,
                                      self.colors[i % len(self.colors)], current_frame, only_arms)
            # self.mark_hand(img, person_path, self.colors[i % len(self.colors)])

        cv2.imshow("output", img)
        cv2.waitKey(15)

    def _add_lines_from_path(self, img, person_path, color, current_frame, only_arms):
        # Don't draw old paths
        if person_path.last_frame_update <= current_frame - 10:
            return

        person_path = person_path.get_keypoint_path(COCOKeypoints.Neck.value)

        start_index = max(1, len(person_path) - 10)
        for i in range(start_index, len(person_path)):
            keypoint = person_path[i].astype(np.int)
            prev_keypoint = person_path[i - 1].astype(np.int)

            # If there are no nonzero keypoints, just move on with your life.
            if any(np.array_equal(k, [0.0, 0.0]) for k in [keypoint, prev_keypoint]):
                continue

            cv2.line(img, tuple(prev_keypoint), tuple(keypoint), color, 3)
