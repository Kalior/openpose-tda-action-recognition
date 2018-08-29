import numpy as np
import scipy.optimize
from time import time
import cv2
import os
import logging

from .track_visualiser import TrackVisualiser
from .person import Person
from .track import Track


class Tracker:
    """Tracks people throgh a video using a detector.

    Parameters
    ----------
    detector : any object implementing detect in the same way as the openpose
        implementations in the detector module.  Used for detecting keypoints
        of people from an image.
    out_dir : str, optional, default 'output'
        path to directory where the resulting tracks and videos are saved.
        Creates this directory if it does not exist.

    """

    def __init__(self, detector, out_dir='output'):
        self.tracks = []

        self.detector = detector

        self.speed_change_threshold = 10

        self.visualiser = TrackVisualiser()

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def video(self, file, draw_frames):
        """Tracks people in the video given in file.

        Produces a video with the identified people overlayed on the
        original video.  Also creates a .npz file with the identified
        tracks of people and the corresponding frame numbers.
        Each track is a [n_frames, n_keypoints, 3], making the final
        outputted array of shape [n_tracks, n_frames, n_keypoints, 3],
        where the values are (x, y, confidence).

        Parameters
        ----------
        file : str
            path to the video for which the tracks should be produced
        draw_frames : boolean
            Specifies if the intermediate frames from the original video
            overlayed with the identified keypoints should be produced
            during computation time or not.

        """
        #  Just loop through the generator as we're only interested
        # in the output at the end.
        for _ in self.video_generator(file, draw_frames):
            continue

        self._save_tracks(file)

    def video_generator(self, file, draw_frames):
        """Tracks people in the video in file, and yields ever frame.

        After each frame, yields the current tracks.  The yielded tracks
        can e.g. be post-processed and actions can be predicted on them.

        Parameters
        ----------
        file : str
            path to the video for which the tracks should be produced
        draw_frames : boolean
            Specifies if the intermediate frames from the original video
            overlayed with the identified keypoints should be produced
            during computation time or not.

        """
        capture = cv2.VideoCapture(file)
        self.speed_change_threshold = 10

        writer = self._create_writer(file, capture)

        current_frame = 0
        success, original_image = capture.read()
        while success:
            track_endpoints = [track.get_last_person()
                               for track in self.tracks
                               if track.recently_updated(current_frame)]

            openpose_start_time = time()
            keypoints, image_with_keypoints = self.detector.detect(original_image)
            people = [p for p in self._convert_to_persons(keypoints)]
            openpose_time = time() - openpose_start_time

            min_person_start_time = time()
            # Find out which people are closest to each other
            assignments, distances, removed_people = self._find_assignments(
                people, track_endpoints, current_frame)

            #  Add back the people we couldn't associate well during the assignment process
            # to the back of the list
            people = people + removed_people

            self._update_tracks(distances, assignments, people, track_endpoints, current_frame)
            closest_person_time = time() - min_person_start_time

            visualisation_start_time = time()
            self.visualiser.draw_tracks(
                self.tracks, image_with_keypoints, current_frame)
            visualisation_time = time() - visualisation_start_time

            if current_frame > 10:
                yield self.tracks, image_with_keypoints, current_frame

            if draw_frames:
                smaller_img = cv2.resize(image_with_keypoints, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("output", smaller_img)
                cv2.waitKey(1)

            # Write the frame to a video
            writer.write(image_with_keypoints)

            logging.debug("OpenPose: {:.5f}, "
                          "Closest person: {:.5f}, "
                          "Draw tracks to img: {:.5f}".format(
                              openpose_time, closest_person_time, visualisation_time))

            success, original_image = capture.read()
            current_frame += 1

        capture.release()
        writer.release()

    def _create_writer(self, in_file, capture):
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))

        basename = os.path.basename(in_file)
        filename, _ = os.path.splitext(basename)
        out_file = os.path.join(self.out_dir, filename + '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        writer = cv2.VideoWriter(out_file, fourcc, fps, (frame_width, frame_height))

        return writer

    def _find_assignments(self, people, prev_people, current_frame):
        # Pre-allocate the distance matrix
        distances = np.empty((len(prev_people), len(people)))

        # And calculate the distances...
        for i, prev_frame_person in enumerate(prev_people):
            for j, person in enumerate(people):
                distance = person.distance(prev_frame_person)
                distances[i, j] = distance

        removed_people = []
        # Find the best assignments between people in the two frames
        valid_assignment = False
        while not valid_assignment:
            assignments = scipy.optimize.linear_sum_assignment(distances)

            valid_assignment, distances, removed_person = self._is_assignment_valid(
                assignments, distances, people, prev_people, current_frame)

            if removed_person is not None:
                removed_people.append(removed_person)

        return assignments, distances, removed_people

    def _is_assignment_valid(self, assignments, distances, people, prev_people, current_frame):
        for from_, to in zip(assignments[0], assignments[1]):
            track_index = prev_people[from_].track_index
            avg_speed = self.tracks[track_index].get_average_speed_in_window(10)
            frames_since_last_update = current_frame - \
                self.tracks[track_index].last_frame_update

            #  If the movement is too large, assume that the new item can't
            # be associated well. (Which will force it to get a new track later
            # in the processing).
            distance_since_last_seen = avg_speed * frames_since_last_update
            if distances[from_, to] > distance_since_last_seen + self.speed_change_threshold:
                logging.debug("Invalid association! from: {}, to: {}, dist: {:.2f}, "
                              "avg_speed: {:.2f}, frames since last update: {}".format(
                                  from_, to, distances[from_, to], avg_speed,
                                  frames_since_last_update))

                distances = np.delete(distances, to, axis=1)
                removed_person = people.pop(to)

                return False, distances, removed_person

        return True, distances, None

    def _update_tracks(self, distances, assignments, people, prev_people, current_frame):
        for from_, to in zip(assignments[0], assignments[1]):
            logging.debug("From: {}, to: {}  people: {}  prev_people: {}".format(
                from_, to, len(people), len(prev_people)))
            track_index = self._establish_index_of_track(from_, to, prev_people, distances)

            people[to].track_index = track_index
            self.tracks[track_index].add_person(people[to], current_frame)

        # If a person is not assigned to a track yet, assign it to a new track
        self._add_unassigned_people(assignments, people, current_frame)

    def _add_unassigned_people(self, assignments, people, current_frame):
        for i, _ in enumerate(people):
            if i not in assignments[1]:
                track = Track()
                people[i].track_index = len(self.tracks)
                track.add_person(people[i], current_frame)
                self.tracks.append(track)

    def _establish_index_of_track(self, from_, to, prev_people, distances):
        # Make sure we know to which track the requested index belongs to
        if from_ < len(prev_people):
            track_index = prev_people[from_].track_index
        else:
            track_index = len(self.tracks)
            self.tracks.append(Track())

        return track_index

    def _convert_to_persons(self, keypoints):
        return [Person(k) for k in keypoints]

    def _save_tracks(self, in_file):
        basename = os.path.basename(in_file)
        filename, _ = os.path.splitext(basename)
        file_path = os.path.join(self.out_dir, filename + '-tracks')

        logging.debug("Creating output tracks.")
        tracks_out = [track.to_np() for track in self.tracks]

        tracks = np.array([p[0] for p in tracks_out], dtype=object)
        frames = np.array([p[1] for p in tracks_out], dtype=object)

        logging.info("Saving tracks to {}".format(file_path))
        np.savez(file_path, tracks=tracks, frames=frames)
