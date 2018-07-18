import cv2
import numpy as np

from util import COCOKeypoints


class TrackVisualiser:

    def __init__(self):
        self.colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                       (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    def draw_video_with_tracks(self, tracks, video, last_frame):
        capture = cv2.VideoCapture(video)

        for i in range(last_frame):
            success, original_image = capture.read()
            self.draw_frame_number(original_image, i)
            self.draw_tracks(tracks, original_image, i)
            cv2.imshow("output", original_image)
            cv2.waitKey(15)

    def draw_tracks(self, tracks, img, current_frame,
                    keypoint_index=COCOKeypoints.Neck.value):
        for i, track in enumerate(tracks):
            track_color = self.colors[i % len(self.colors)]
            self._add_lines_from_track(img, track, track_color,
                                       current_frame, keypoint_index)
            self._add_index_of_track(img, i, track, track_color, current_frame, keypoint_index)

    def draw_people(self, tracks, img, current_frame, offset_person=True):
        for i, track in enumerate(tracks):
            track_color = self.colors[i % len(self.colors)]
            positions = [(0, 0)] * 15
            for i in range(15):
                path = track.get_keypoint_path(i, current_frame)
                if len(path) > 0:
                    original_pos = path[-1].astype(np.int)
                    if offset_person:
                        offset = np.array([250, 150])
                        position = tuple(original_pos + offset)
                    else:
                        position = tuple(original_pos)
                    cv2.circle(img, position, 5, track_color, 3)
                    positions[i] = position

            connections = [
                (0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
                (1, 8), (1, 11), (8, 9), (9, 10), (11, 12), (12, 13)
            ]
            for from_, to in connections:
                self._add_line(img, positions[from_], positions[to], track_color)

    def _add_line(self, img, from_, to, color):
        if all((0, 0) != p for p in [from_, to]):
            cv2.line(img, from_, to, color, 3)

    def draw_frame_number(self, img, current_frame):
        white = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(current_frame), (50, 50), font, 2, white, 2)

    def draw_text(self, img, text, position=(50, 50)):
        white = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, position, font, 2, white, 2)

    def _add_index_of_track(self, img, track_index, track, color, current_frame, keypoint_index):
        if track.last_frame_update <= current_frame - 10:
            return

        path = track.get_keypoint_path(keypoint_index, current_frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        path_index = max(0, len(path) - 11)
        if len(path) > 0:
            keypoint = path[path_index].astype(np.int)
            cv2.putText(img, str(track_index), tuple(keypoint), font, 4, color, 3)

    def _add_lines_from_track(self, img, track, color, current_frame, keypoint_index):
        # Don't draw old paths
        if track.last_frame_update <= current_frame - 10:
            return

        path = track.get_keypoint_path(keypoint_index, current_frame)

        self._draw_path(img, path, color)

    def _draw_path(self, img, path, color):
        start_index = max(1, len(path) - 10)
        for i in range(start_index, len(path)):
            keypoint = path[i].astype(np.int)
            prev_keypoint = path[i - 1].astype(np.int)

            cv2.line(img, tuple(prev_keypoint), tuple(keypoint), color, 3)
