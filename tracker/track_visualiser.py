import cv2
import numpy as np

from util import COCOKeypoints


class TrackVisualiser:

    def __init__(self):
        self.colors = [(255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]

    def draw_video_with_tracks(self, tracks, video, last_frame):
        capture = cv2.VideoCapture(video)

        for i in range(last_frame):
            success, original_image = capture.read()
            visualiser.draw_frame_number(original_image, i)
            visualiser.draw_tracks(tracks, original_image, i)
            cv2.imshow("output", original_image)
            cv2.waitKey(15)

    def draw_tracks(self, tracks, img, current_frame, only_track_arms=False):
        for i, track in enumerate(tracks):
            track_color = self.colors[i % len(self.colors)]
            self._add_lines_from_track(img, track, track_color, current_frame, only_track_arms)
            self._add_index_of_track(img, i, track, track_color, current_frame)

    def draw_frame_number(self, img, current_frame):
        black = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(current_frame), (50, 50), font, 2, black, 2)

    def _add_index_of_track(self, img, track_index, track, color, current_frame):
        if track.last_frame_update <= current_frame - 10:
            return

        path = track.get_keypoint_path(COCOKeypoints.Neck.value, current_frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        path_index = max(0, len(path) - 11)
        if len(path) > 0:
            keypoint = path[path_index].astype(np.int)
            cv2.putText(img, str(track_index), tuple(keypoint), font, 4, color)

    def _add_lines_from_track(self, img, track, color, current_frame, only_track_arms):
        # Don't draw old paths
        if track.last_frame_update <= current_frame - 10:
            return

        if only_track_arms:
            path = track.get_keypoint_path(COCOKeypoints.RWrist.value, current_frame)
        else:
            path = track.get_keypoint_path(COCOKeypoints.Neck.value, current_frame)

        self._draw_path(img, path, color)

    def _draw_path(self, img, path, color):
        start_index = max(1, len(path) - 10)
        for i in range(start_index, len(path)):
            keypoint = path[i].astype(np.int)
            prev_keypoint = path[i - 1].astype(np.int)

            cv2.line(img, tuple(prev_keypoint), tuple(keypoint), color, 3)
