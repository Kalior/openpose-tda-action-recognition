import cv2
import numpy as np

from util import COCOKeypoints


class TrackVisualiser(object):

    def __init__(self):
        self.colors = [(255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]

    def draw_tracks(self, tracks, img, current_frame, only_track_arms=False):
        for i, track in enumerate(tracks):
            self._add_lines_from_track(img, track, self.colors[i % len(self.colors)],
                                       current_frame, only_track_arms)

        cv2.imshow("output", img)
        cv2.waitKey(1)

    def _add_lines_from_track(self, img, track, color, current_frame, only_track_arms):
        # Don't draw old paths
        if track.last_frame_update <= current_frame - 10:
            return

        if only_track_arms:
            path = track.get_keypoint_path(COCOKeypoints.RWrist.value, current_frame)
        else:
            path = track.get_keypoint_path(COCOKeypoints.Neck.value, current_frame)

        start_index = max(1, len(path) - 10)
        for i in range(start_index, len(path)):
            keypoint = path[i].astype(np.int)
            prev_keypoint = path[i - 1].astype(np.int)

            cv2.line(img, tuple(prev_keypoint), tuple(keypoint), color, 3)
