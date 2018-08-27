import cv2
import numpy as np

from ..util import COCOKeypoints, coco_connections


class TrackVisualiser:
    """Helper class which uses opencv to draw videos with various overlays.

    """

    def __init__(self):
        self.colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                       (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    def draw_video_with_tracks(self, tracks, video, last_frame, start_frame=0):
        """Draws the video from start_frame to last_frame with the tracks overlayed.

        Parameters
        ----------
        tracks : list of Track
            The tracks which should be overlayed on the video.
        video : str
            Path to the video from which the tracks were produced.
        last_frame : int
            The last frame that should be drawn.
        start_frame : int
            The start frame of the visualisation.
        """
        capture = cv2.VideoCapture(video)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for i in range(start_frame, last_frame):
            success, original_image = capture.read()
            self.draw_frame_number(original_image, i)
            self.draw_people(tracks, original_image, i, False)

            smaller_original = cv2.resize(original_image, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("output", smaller_original)
            cv2.waitKey(10)

    def draw_tracks(self, tracks, img, current_frame, keypoint_index=COCOKeypoints.Neck.value):
        """Overlays the tracks on the img.

        Parameters
        ----------
        tracks : list of Track
        img : array-like
            The image to overlay the tracks on.
        current_frame : int
            The current frame to get the keypoints from tracks for.
        keypoint_index : int, optional, default 1
            Specifies which keypoint should be drawn to the image.

        """
        for i, track in enumerate(tracks):
            track_color = self.colors[i % len(self.colors)]
            self._add_lines_from_track(img, track, track_color,
                                       current_frame, keypoint_index)
            self._add_index_of_track(img, i, track, track_color, current_frame, keypoint_index)

    def draw_people(self, tracks, img, current_frame, offset_person=True):
        """Overlays the skeleton of people from tracks to image.

        Parameters
        ----------
        tracks : list of Track
        img : array-like
            The image to overlay the tracks on.
        current_frame : int
            The current frame to get the keypoints from tracks for.
        offset_person : boolean, optional, default True
            Specifies if the person should be offsettted from origin.

        """
        for i, track in enumerate(tracks):
            track_color = self.colors[i % len(self.colors)]
            positions = [(0, 0)] * 14
            keypoints = track.get_keypoints_at(current_frame)
            for i in range(14):
                original_pos = keypoints[i].astype(np.int)
                if offset_person:
                    offset = np.array([250, 150])
                    position = tuple(original_pos + offset)
                else:
                    position = tuple(original_pos)
                cv2.circle(img, position, 5, track_color, 3)
                positions[i] = position

            for from_, to in coco_connections:
                self._add_line(img, positions[from_], positions[to], track_color)

    def _add_line(self, img, from_, to, color):
        if all((0, 0) != p for p in [from_, to]):
            cv2.line(img, from_, to, color, 3)

    def draw_frame_number(self, img, current_frame, color=(255, 255, 255)):
        """Overlays the frame number on the img.

        Parameters
        ----------
        img : array-like
            The image to overlay the current frame index on.
        current_frame : int
            The int to overlay on the frame.
        color : triple (3-tuple) of int
            The color of the number.

        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.draw_text(img, str(current_frame), position=(50, 50), color=color)

    def draw_text(self, img, text, position=(50, 50), color=(255, 255, 255)):
        """Overlays the text on the img at the position.

        Parameters
        ----------
        img : array-like
            The image to overlay the current frame index on.
        text : str
            The text to overlay on the frame.
        position : tuple of int
            The position where the text should be drawn.
        color : triple (3-tuple) of int
            The color of the number.

        """
        black = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, position, font, 2, black, 4)
        cv2.putText(img, text, position, font, 2, color, 2)

    def _add_index_of_track(self, img, track_index, track, color, current_frame, keypoint_index):
        if track.last_frame_update <= current_frame - 10:
            return

        path = track.get_keypoint_path(keypoint_index, current_frame)

        if len(path) > 0:
            keypoint = path[-1].astype(np.int)
            self.draw_text(img, str(track_index), position=tuple(keypoint), color=color)

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
